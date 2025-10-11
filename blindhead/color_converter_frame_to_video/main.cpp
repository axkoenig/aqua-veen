#include <iostream>
#include <stdexcept>
#include <string>
#include <array>
#include <cstdlib>

// FFmpeg headers
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

// Default color definitions in RGB (from hex strings)
static const std::array<uint8_t, 3> DEFAULT_BACKGROUND_COLOR = {0x1f, 0x24, 0x36}; // "#1f2436"
static const std::array<uint8_t, 3> DEFAULT_POSITIVE_COLOR   = {0xfb, 0x37, 0x37}; // "#fb3737"
static const std::array<uint8_t, 3> DEFAULT_NEGATIVE_COLOR   = {0xfd, 0xf2, 0xf2}; // "#fdf2f2"

// Function to convert hex string (e.g., "#fb3737") into an RGB array.
std::array<uint8_t, 3> hexToRGB(const std::string &hex) {
    std::string s = hex;
    if (s[0] == '#')
        s = s.substr(1);
    if (s.size() != 6) {
        throw std::invalid_argument("Invalid hex color format: " + hex);
    }
    uint8_t r = static_cast<uint8_t>(std::stoul(s.substr(0, 2), nullptr, 16));
    uint8_t g = static_cast<uint8_t>(std::stoul(s.substr(2, 2), nullptr, 16));
    uint8_t b = static_cast<uint8_t>(std::stoul(s.substr(4, 2), nullptr, 16));
    return { r, g, b };
}

// Function to process an RGB image buffer based on grayscale threshold.
// For each pixel the average value (R+G+B)/3 is calculated, and then:
//   if intensity <= 70, use the background color,
//   else if intensity < 200, use the positive color,
//   else (>=200) use the negative color.
void convert_colors(uint8_t* data, int width, int height, int linesize,
                    const uint8_t background_color[3],
                    const uint8_t positive_color[3],
                    const uint8_t negative_color[3],
                    int background_max_int = 70, int positive_min_int = 200)
{
    for (int j = 0; j < height; j++) {
        uint8_t* row = data + j * linesize;
        for (int i = 0; i < width; i++) {
            uint8_t* pixel = row + i * 3; // RGB24: 3 bytes per pixel
            int gray = (pixel[0] + pixel[1] + pixel[2]) / 3;
            if (gray <= background_max_int) {
                pixel[0] = background_color[0];
                pixel[1] = background_color[1];
                pixel[2] = background_color[2];
            } else if (gray < positive_min_int) {
                pixel[0] = positive_color[0];
                pixel[1] = positive_color[1];
                pixel[2] = positive_color[2];
            } else { // gray >= positive_min_int
                pixel[0] = negative_color[0];
                pixel[1] = negative_color[1];
                pixel[2] = negative_color[2];
            }
        }
    }
}

// Process a single video: open input file, decode frame by frame, process and encode, print progress.
// A new parameter "crf_value" is added for video compression quality (default "18"),
// and a new boolean parameter "upscale4k" enables export at 3840x2160 using NN (SWS_POINT) upsampling.
int process_video(const std::string &input_filename,
                  const std::string &output_filename,
                  const std::array<uint8_t, 3> &background_color,
                  const std::array<uint8_t, 3> &positive_color,
                  const std::array<uint8_t, 3> &negative_color,
                  const std::string &crf_value = "18",
                  bool upscale4k = false)
{
    int ret;
    AVDictionary *fmt_opts = nullptr;
    // Set flag to generate PTS even if the index is broken/missing
    av_dict_set(&fmt_opts, "fflags", "genpts", 0);

    // Open input file as a stream
    AVFormatContext *in_fmt_ctx = nullptr;
    if ((ret = avformat_open_input(&in_fmt_ctx, input_filename.c_str(), nullptr, &fmt_opts)) < 0) {
        std::cerr << "Could not open input file: " << input_filename << std::endl;
        return ret;
    }
    av_dict_free(&fmt_opts);

    // Retrieve stream information
    if ((ret = avformat_find_stream_info(in_fmt_ctx, nullptr)) < 0) {
        std::cerr << "Failed to retrieve input stream information" << std::endl;
        return ret;
    }

    // Find the first video stream
    int video_stream_index = -1;
    for (unsigned int i = 0; i < in_fmt_ctx->nb_streams; i++) {
        if (in_fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1) {
        std::cerr << "No video stream found in " << input_filename << std::endl;
        return -1;
    }

    // Set up the decoder for the video stream
    AVCodecParameters *codecpar = in_fmt_ctx->streams[video_stream_index]->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(codecpar->codec_id);
    if (!decoder) {
        std::cerr << "Decoder not found" << std::endl;
        return -1;
    }
    AVCodecContext *dec_ctx = avcodec_alloc_context3(decoder);
    if (!dec_ctx) {
        std::cerr << "Could not allocate decoder context" << std::endl;
        return AVERROR(ENOMEM);
    }
    if ((ret = avcodec_parameters_to_context(dec_ctx, codecpar)) < 0) {
        std::cerr << "Failed to copy decoder parameters to context" << std::endl;
        return ret;
    }
    if ((ret = avcodec_open2(dec_ctx, decoder, nullptr)) < 0) {
        std::cerr << "Failed to open decoder" << std::endl;
        return ret;
    }

    // Prepare the output format context. If the output filename ends with ".mp4", an MP4 container will be used.
    AVFormatContext *out_fmt_ctx = nullptr;
    if ((ret = avformat_alloc_output_context2(&out_fmt_ctx, nullptr, nullptr, output_filename.c_str())) < 0) {
        std::cerr << "Could not create output context" << std::endl;
        return ret;
    }

    // Create a new video stream for output
    AVStream *out_stream = avformat_new_stream(out_fmt_ctx, nullptr);
    if (!out_stream) {
        std::cerr << "Failed allocating output stream" << std::endl;
        return AVERROR_UNKNOWN;
    }

    // For the encoder, we choose H.264 (libx264 if available)
    const AVCodec *encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!encoder) {
        std::cerr << "Necessary encoder not found" << std::endl;
        return AVERROR_INVALIDDATA;
    }
    AVCodecContext *enc_ctx = avcodec_alloc_context3(encoder);
    if (!enc_ctx) {
        std::cerr << "Could not allocate an encoding context" << std::endl;
        return AVERROR(ENOMEM);
    }

    // Set basic encoder parameters (using values from the input video)
    enc_ctx->height = dec_ctx->height;
    enc_ctx->width  = dec_ctx->width;
    enc_ctx->sample_aspect_ratio = dec_ctx->sample_aspect_ratio;
    // If 4K export is enabled, override with 3840x2160.
    if (upscale4k) {
        enc_ctx->width = 3840;
        enc_ctx->height = 2160;
    }
    // We want output in YUV420P (which the H.264 encoder expects)
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    // Set the time base. Prefer the average frame rate if available.
    if (in_fmt_ctx->streams[video_stream_index]->avg_frame_rate.num != 0)
        enc_ctx->time_base = av_inv_q(in_fmt_ctx->streams[video_stream_index]->avg_frame_rate);
    else
        enc_ctx->time_base = (AVRational){1, 30};

    // Optionally set additional encoder parameters (bitrate, etc.)
    enc_ctx->bit_rate = 4000000; // 4 Mbps as an example

    // Expose the CRF value for compression quality.
    // CRF values range from 0 (lossless) to 51 (worst); 18 is a common choice for high quality.
    if(enc_ctx->priv_data) {
        av_opt_set(enc_ctx->priv_data, "crf", crf_value.c_str(), 0);
    }

    // Open the encoder
    if ((ret = avcodec_open2(enc_ctx, encoder, nullptr)) < 0) {
        std::cerr << "Cannot open video encoder for stream" << std::endl;
        return ret;
    }

    // Copy encoder settings to the output stream
    ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
    if (ret < 0) {
        std::cerr << "Failed to copy encoder parameters to output stream" << std::endl;
        return ret;
    }
    out_stream->time_base = enc_ctx->time_base;

    // Open the output file (if needed)
    if (!(out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if ((ret = avio_open(&out_fmt_ctx->pb, output_filename.c_str(), AVIO_FLAG_WRITE)) < 0) {
            std::cerr << "Could not open output file " << output_filename << std::endl;
            return ret;
        }
    }

    // Write the stream header to the output file.
    if ((ret = avformat_write_header(out_fmt_ctx, nullptr)) < 0) {
        std::cerr << "Error occurred when opening output file" << std::endl;
        return ret;
    }

    // Allocate necessary frames
    AVFrame *in_frame = av_frame_alloc();
    if (!in_frame) {
        std::cerr << "Could not allocate frame" << std::endl;
        return AVERROR(ENOMEM);
    }
    // Allocate frame for the intermediate RGB24 image.
    AVFrame *rgb_frame = av_frame_alloc();
    if (!rgb_frame) {
        std::cerr << "Could not allocate RGB frame" << std::endl;
        return AVERROR(ENOMEM);
    }
    rgb_frame->width = dec_ctx->width;
    rgb_frame->height = dec_ctx->height;
    rgb_frame->format = AV_PIX_FMT_RGB24;
    ret = av_image_alloc(rgb_frame->data, rgb_frame->linesize, rgb_frame->width,
                         rgb_frame->height, AV_PIX_FMT_RGB24, 32);
    if (ret < 0) {
        std::cerr << "Could not allocate raw picture buffer" << std::endl;
        return ret;
    }
    // Allocate frame to send to encoder (in YUV420P)
    AVFrame *enc_frame = av_frame_alloc();
    if (!enc_frame) {
        std::cerr << "Could not allocate encoder frame" << std::endl;
        return AVERROR(ENOMEM);
    }
    enc_frame->width  = enc_ctx->width;
    enc_frame->height = enc_ctx->height;
    enc_frame->format = enc_ctx->pix_fmt;
    ret = av_frame_get_buffer(enc_frame, 32);
    if (ret < 0) {
        std::cerr << "Could not allocate the video frame data" << std::endl;
        return ret;
    }

    // Create two scaling contexts:
    // 1. From the decoder's pixel format to RGB24 (for our custom processing).
    AVPixelFormat input_pix_fmt = dec_ctx->pix_fmt;
    if (input_pix_fmt == AV_PIX_FMT_YUVJ420P)
        input_pix_fmt = AV_PIX_FMT_YUV420P;
    struct SwsContext* sws_ctx_to_rgb = sws_getContext(dec_ctx->width, dec_ctx->height, input_pix_fmt,
                                                         dec_ctx->width, dec_ctx->height, AV_PIX_FMT_RGB24,
                                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_to_rgb) {
        std::cerr << "Could not initialize the conversion context to RGB" << std::endl;
        return AVERROR_UNKNOWN;
    }
    // 2. From the processed RGB24 to the encoder's pixel format (YUV420P)
    // Use nearest-neighbor upsampling (SWS_POINT) if 4K export is enabled.
    int scaling_flags = upscale4k ? SWS_POINT : SWS_BILINEAR;
    struct SwsContext* sws_ctx_to_yuv = sws_getContext(rgb_frame->width, rgb_frame->height, AV_PIX_FMT_RGB24,
                                                         enc_ctx->width, enc_ctx->height, enc_ctx->pix_fmt,
                                                         scaling_flags, nullptr, nullptr, nullptr);
    if (!sws_ctx_to_yuv) {
        std::cerr << "Could not initialize the conversion context to YUV" << std::endl;
        return AVERROR_UNKNOWN;
    }

    // Allocate packet structures
    AVPacket *packet = av_packet_alloc();
    AVPacket *enc_pkt = av_packet_alloc();
    if (!packet || !enc_pkt) {
        std::cerr << "Could not allocate AVPacket" << std::endl;
        return AVERROR(ENOMEM);
    }

    // Estimate total frames if possible (nb_frames isn't always set)
    int64_t total_frames = in_fmt_ctx->streams[video_stream_index]->nb_frames;
    if(total_frames == 0 && in_fmt_ctx->streams[video_stream_index]->duration > 0) {
        AVRational avg_frame_rate = in_fmt_ctx->streams[video_stream_index]->avg_frame_rate;
        total_frames = in_fmt_ctx->streams[video_stream_index]->duration * avg_frame_rate.num / avg_frame_rate.den;
    }
    int frame_count = 0;

    // Read and process frames packet by packet.
    while (av_read_frame(in_fmt_ctx, packet) >= 0) {
        if(packet->stream_index != video_stream_index) {
            av_packet_unref(packet);
            continue;
        }
        ret = avcodec_send_packet(dec_ctx, packet);
        if (ret < 0) {
            std::cerr << "Error sending packet for decoding" << std::endl;
            break;
        }
        // Receive all available frames from the decoder
        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx, in_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            else if (ret < 0) {
                std::cerr << "Error during decoding" << std::endl;
                goto cleanup;
            }

            // Convert the decoded frame to RGB24.
            sws_scale(sws_ctx_to_rgb,
                      in_frame->data, in_frame->linesize,
                      0, dec_ctx->height,
                      rgb_frame->data, rgb_frame->linesize);

            // Apply our custom color conversion on the RGB data.
            convert_colors(rgb_frame->data[0], rgb_frame->width, rgb_frame->height,
                           rgb_frame->linesize[0],
                           background_color.data(), positive_color.data(), negative_color.data());

            // Convert the processed RGB frame to the encoder pixel format (YUV420P) for output.
            sws_scale(sws_ctx_to_yuv,
                      rgb_frame->data, rgb_frame->linesize,
                      0, rgb_frame->height,
                      enc_frame->data, enc_frame->linesize);

            // Set presentation timestamp (pts) for the encoder.
            enc_frame->pts = in_frame->pts;

            // Send the frame to the encoder.
            ret = avcodec_send_frame(enc_ctx, enc_frame);
            if (ret < 0) {
                std::cerr << "Error sending frame for encoding" << std::endl;
                goto cleanup;
            }
            // Retrieve encoded packets
            while (ret >= 0) {
                ret = avcodec_receive_packet(enc_ctx, enc_pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                else if (ret < 0) {
                    std::cerr << "Error during encoding" << std::endl;
                    goto cleanup;
                }
                enc_pkt->stream_index = out_stream->index;
                // Rescale packet timestamp values from encoder time_base to output stream time_base
                av_packet_rescale_ts(enc_pkt, enc_ctx->time_base, out_stream->time_base);
                ret = av_interleaved_write_frame(out_fmt_ctx, enc_pkt);
                if (ret < 0) {
                    std::cerr << "Error during writing packet" << std::endl;
                    goto cleanup;
                }
                av_packet_unref(enc_pkt);
            }

            frame_count++;

            // Print progress bar (if total_frames is known, otherwise show frame count).
            if(total_frames > 0) {
                float progress = (float)frame_count / total_frames;
                int bar_width = 50;
                std::cout << "\r[";
                int pos = static_cast<int>(bar_width * progress);
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << " %" << std::flush;
            } else {
                std::cout << "\rFrame: " << frame_count << std::flush;
            }

            av_frame_unref(in_frame);
        }
        av_packet_unref(packet);
    }
    std::cout << std::endl;

    // Flush the encoder
    ret = avcodec_send_frame(enc_ctx, nullptr);
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, enc_pkt);
        if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
            break;
        else if (ret < 0) {
            std::cerr << "Error during encoding flushing" << std::endl;
            goto cleanup;
        }
        enc_pkt->stream_index = out_stream->index;
        av_packet_rescale_ts(enc_pkt, enc_ctx->time_base, out_stream->time_base);
        ret = av_interleaved_write_frame(out_fmt_ctx, enc_pkt);
        if (ret < 0)
            break;
        av_packet_unref(enc_pkt);
    }

    // Write the trailer to the output file.
    av_write_trailer(out_fmt_ctx);
    
    // Log that the video was successfully saved.
    std::cout << "Video saved to: " << output_filename << std::endl;

cleanup:
    // Free allocated resources
    av_packet_free(&packet);
    av_packet_free(&enc_pkt);
    av_frame_free(&in_frame);
    if(rgb_frame) {
        av_freep(&rgb_frame->data[0]);
        av_frame_free(&rgb_frame);
    }
    av_frame_free(&enc_frame);
    sws_freeContext(sws_ctx_to_rgb);
    sws_freeContext(sws_ctx_to_yuv);
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    if (in_fmt_ctx)
        avformat_close_input(&in_fmt_ctx);
    if (out_fmt_ctx && !(out_fmt_ctx->oformat->flags & AVFMT_NOFILE))
        avio_closep(&out_fmt_ctx->pb);
    avformat_free_context(out_fmt_ctx);

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video> [--background_color #xxxxxx] "
                  << "[--positive_color #xxxxxx] [--negative_color #xxxxxx] [--crf <value>] [--4k]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string input_filename  = argv[1];
    std::string output_filename = argv[2];
    std::string background_color_str = "#1f2436";
    std::string positive_color_str   = "#fb3737";
    std::string negative_color_str   = "#fdf2f2";
    std::string crf_value = "18";
    bool upscale4k = false;

    // Process any optional arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--background_color" && i + 1 < argc) {
            background_color_str = argv[++i];
        } else if (arg == "--positive_color" && i + 1 < argc) {
            positive_color_str = argv[++i];
        } else if (arg == "--negative_color" && i + 1 < argc) {
            negative_color_str = argv[++i];
        } else if (arg == "--crf" && i + 1 < argc) {
            crf_value = argv[++i];
        } else if (arg == "--4k") {
            upscale4k = true;
        }
    }

    std::array<uint8_t, 3> background_color;
    std::array<uint8_t, 3> positive_color;
    std::array<uint8_t, 3> negative_color;
    try {
        background_color = hexToRGB(background_color_str);
        positive_color   = hexToRGB(positive_color_str);
        negative_color   = hexToRGB(negative_color_str);
    } catch(const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Process the video file.
    if (process_video(input_filename, output_filename, background_color, positive_color, negative_color, crf_value, upscale4k) < 0) {
        std::cerr << "Error processing video" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
