// main.cpp
// A C++ version of realtime 3D event visualization with event loading from HDF5.
// Dependencies: GLFW, GLEW, GLM, HDF5 C++ API (dummy load here)
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <array>
#include <memory>
#include <cstdio>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <hdf5.h>
#include <filesystem>      // Added for obtaining absolute file paths (C++17)
#include <unistd.h>        // Added for getpid()
#include <limits>          // Added for std::numeric_limits
#include <cstring>         // Added for strncpy

// Include OpenGL and GLFW headers.
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// For matrix math.
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Add these includes at the top with the other includes
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// Include the nlohmann/json header (make sure to have the header in your include paths)
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ---------------- Screen Layout Configuration ---------------- //

// Set screen layout option: vertical (portrait) or horizontal (landscape).
// When vertical, the layout is 720p wide (720 x 1280).
const bool g_isVerticalLayout = true;  // Change to 'true' for vertical layout

// Event Volume Parameters (dimensions change based on layout selection)
const float EVENT_VOLUME_WIDTH  = g_isVerticalLayout ? 720.0f  : 1280.0f;
const float EVENT_VOLUME_HEIGHT = g_isVerticalLayout ? 1280.0f : 720.0f;

// Configuration Parameters
const int g_GUI_FIXED_WIDTH = 400;           // Global variable for fixed GUI panel width.

// When creating the window, update WINDOW_WIDTH as follows:
const int WINDOW_WIDTH = static_cast<int>(EVENT_VOLUME_WIDTH) + g_GUI_FIXED_WIDTH;
const int WINDOW_HEIGHT = EVENT_VOLUME_HEIGHT;

// Camera Parameters
const float ROTATION_SENSITIVITY = 0.25f;
const float TRANSLATION_SENSITIVITY = 1.0f;
const float MIN_CAMERA_DISTANCE = 1.0f;
const float ZOOM_SENSITIVITY = 20.0f;
const float MAX_ELEVATION = 89.0f;

// Coordinate Axis Parameters
const float AXIS_LENGTH = 100.0f;
const float AXIS_LINE_WIDTH = 7.0f;

// Simulation Parameters
const float INITIAL_WINDOW_SIZE = 100.0f;  // Initial Z length
const float MAX_WINDOW_SIZE = 10000.0f;     // Maximum Z length
const float MIN_WINDOW_SIZE = 10.0f;       // Minimum Z length

// Point Rendering Parameters
const float INITIAL_POINT_SIZE = 3.0f;
const float MIN_POINT_SIZE = 1.0f;
const float MAX_POINT_SIZE = 20.0f;

// Slow Motion Factors
const float MIN_SLOMO_FACTOR = 0.001f;
const float MAX_SLOMO_FACTOR = 2.0f;
const float INITIAL_SLOMO_FACTOR = 1.0f;

// Add this new constant for fixed buffer size:
const size_t FIXED_BUFFER_SIZE = 1000000;

// ---------------- Global Variables ---------------- //

// Event Loading Variables
std::string g_eventFilePath;
std::atomic<bool> g_shouldStopLoading{false};
float g_timeWindow = 0.1f; // Default time window in seconds

// Event Structure and Containers
struct Event {
    float x, y, t, p;
};
std::vector<Event> g_events;
std::atomic<bool> g_eventsLoaded{false};
std::vector<Event> g_prefetchedEvents;

// Global HDF5 mutex declared somewhere in your common header or at the top of this file.
std::mutex g_hdf5Mutex;

// Simulation State
float g_windowSize = INITIAL_WINDOW_SIZE;
float g_simTimeMicroSeconds = 0.0f; // simulation time in microseconds
float g_simTimeMicroSecondsTask = 0.0f; // for manually setting the visualization start time
float g_startTimeMicroSeconds = 0.0f;
float g_endTimeMicroSeconds = 0.0f;
bool g_paused = false;

// Global variables for event time range
float g_minEventTime = 0.0f;
float g_maxEventTime = 0.0f;

// Add this near the other global variables
bool g_render4K = true;  // Option to render in 4K resolution

// Camera
struct Camera {
    float elevation = 0.0f;
    float azimuth = 0.0f;
    float distance = 500.0f;
    glm::vec3 center = glm::vec3(0.0f);
};
Camera g_camera;

// Add this enum (e.g., near your Camera definition)
enum class TaskType {
    CameraMove,
    SetSimTime,
    Wait,
    SetZLength,   // For Z length transition.
    CameraRotate, // For smooth rotation around a fixed point.
    SetSlomoFactor // New: to smoothly adjust the slomo factor.
};

// UPDATED: The CameraTask now carries a type and a simTime field.
struct CameraTask {
    TaskType type = TaskType::CameraMove; // default is CameraMove
    Camera camera;                      // used if type is CameraMove
    std::string name;
    float duration = 3.0f;              // Duration (in seconds) for the transition
    
    // Only applicable if type is SetSimTime.
    float simTime = 0.0f;

    // For SetZLength: target and (to be captured) start window size.
    float targetWindowSize = 0.0f;
    float startWindowSize = 0.0f;

    // For CameraRotate.
    glm::vec3 rotationAxis = glm::vec3(0.0f, 1.0f, 0.0f); 
    float rotationAngle = 90.0f;

    // ---- New: Field for slow-motion factor.
    float targetSlomoFactor = INITIAL_SLOMO_FACTOR; // The target slomo factor.
};

// Global counter for assigning unique (and persistent) names.
int g_cameraTaskCounter = 1;

// Simulation Task Variables
bool g_runningTasks = false;
std::vector<CameraTask> g_cameraTasks;
size_t g_currentTaskIndex = 0;
float g_taskTimeAccumulator = 0.0f;

// Variables used for an individual task run.
bool g_individualTaskRun = false;
std::vector<Camera> g_individualTrajectory;

// Timing
double g_lastTimeSeconds = 0.0;

// Point Size and Colors
float g_pointSize = INITIAL_POINT_SIZE;
glm::vec4 g_positiveEventColor(0.984313725490196f, 0.21568627450980393f, 0.21568627450980393f, 1.0f);
glm::vec4 g_negativeEventColor(0.9921568627450981f, 0.9490196078431372f, 0.9490196078431372f, 1.0f);
glm::vec4 g_backgroundColor(0.0f, 0.0f, 0.0f, 1.0f);

// New global flag to toggle the volume box visibility.
bool g_showVolumeBox = true;

// New global flag to toggle the coordinate axes visibility.
bool g_showCoordinateAxes = false;

// Color Randomization Globals
bool g_randomizeColorsActive = false;
float g_colorTransitionDuration = 1.0f; // Duration (in seconds) for each color interpolation (now mutable)
float g_colorTransitionTimer = 0.0f;
glm::vec4 g_positiveEventStartColor = g_positiveEventColor;
glm::vec4 g_negativeEventStartColor = g_negativeEventColor;
glm::vec4 g_positiveEventTargetColor = g_positiveEventColor;
glm::vec4 g_negativeEventTargetColor = g_negativeEventColor;

// Shader Sources
const char* vertexShaderSource = R"(
#version 150 core
in vec3 position;
in vec4 in_color;
uniform mat4 MVP;
uniform float pointSize;
out vec4 fragColor;
void main() {
    gl_Position = MVP * vec4(position, 1.0);
    fragColor = in_color;
    gl_PointSize = pointSize;
}
)";

const char* fragmentShaderSource = R"(
#version 150 core
in vec4 fragColor;
out vec4 finalColor;
void main() {
    finalColor = fragColor;
}
)";

// Continuous Event Loading & Prefetching
std::mutex g_eventsMutex;
std::atomic<bool> g_stopLoading{false}; // Loader thread stop signal.
int g_maxDisplayEvents = 15000;
float g_slomoFactor = INITIAL_SLOMO_FACTOR;
float g_slomoFactorSmooth = INITIAL_SLOMO_FACTOR;  // NEW: Used to smooth changes in slomo factor.
int g_volumeRotationIndex = 0; // 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270° rotation around Z axis.

// Mouse State
bool g_isDragging = false;
double g_lastX = 0.0, g_lastY = 0.0;

// Frame Timing and FPS
double g_lastFrameTime = 0.0;
double g_fps = 0.0;
const int FPS_SAMPLE_COUNT = 60;
std::array<double, FPS_SAMPLE_COUNT> g_fpsHistory{};
int g_fpsIndex = 0;

// Coordinate Axes Buffers
GLuint g_axisVAO, g_axisVBO, g_axisColorVBO;

// System Monitoring Variables
struct SystemStats {
    std::string cpuUsage = "0%";
    std::string ramUsage = "0%";
    std::string gpuUsage = "0%";
};
SystemStats g_systemStats;
std::mutex g_statsMutex;
std::atomic<bool> g_shouldStopMonitoring{false};

// Box (Event Volume) Buffers
GLuint g_boxVAO, g_boxVBO, g_boxColorVBO;

// Number of Points Rendered
size_t g_numPoints = 0;

// Global variables for individual task execution (for both CameraMove and SetSimTime).
TaskType g_individualTaskType = TaskType::CameraMove;
float g_individualSegmentDuration = 2.0f; // already defined previously for camera moves
float g_initialSimTime = 0.0f;  // for interpolating simulation time
float g_targetSimTime = 0.0f;   // target simulation time for a SetSimTime task

// Global variable to store the starting camera pose for full-run mode.
Camera g_startCamera;

// Video Rendering Globals
bool g_renderMode = false;                         // true when a full render (recording) is requested
bool g_isRecording = false;                        // true while frames are being captured
std::atomic<bool> g_videoSaving {false};           // true when video is being written to disk
double g_videoProgress = 0.0;                      // fraction [0,1] of writing process completed
int g_videoWidth = 0;                              // will be set from the viewport dimensions
int g_videoHeight = 0;

// Global variable for task list name input in ImGui.
char g_taskListNameBuffer[256] = "DefaultTaskList";

// Add near the other global variables.
float g_individualStartWindowSize = 0.0f;
float g_individualTargetWindowSize = 0.0f;

// Global variable to capture the starting slomo factor for individual SetSlomoFactor tasks.
float g_individualStartSlomo = 0.0f;

// Add these new globals near the other global variables
std::string g_dataDir = "data";
std::vector<std::string> g_hdf5Files;
int g_selectedFileIndex = -1;

// Add this near the other global variables at the top of the file
std::thread g_loaderThread;

// Global event buffer handles
GLuint g_eventVBO = 0;
GLuint g_eventCBO = 0;

// Global flag:
bool g_shouldClearBuffers = false;

// Add this as a global variable with the other OpenGL-related globals
bool g_buffersInitialized = false;

// Add to global variables section
bool g_showEventGraph = false;         // Toggle for graph visualization
float g_graphMaxDistance = 50.0f;      // Max distance for creating connections
int g_graphMaxConnections = 3;         // Max connections per event
float g_graphLineWidth = 0.5f;         // Width of graph lines
float g_graphConnectionProb = 0.2f;    // Probability of creating a connection (0.0-1.0)
glm::vec4 g_graphLineColor = glm::vec4(0.5f, 0.8f, 1.0f, 0.3f);  // Color of graph lines
bool g_connectPositiveOnly = false;    // If true, only connect positive events
bool g_connectSamePolarity = true;     // If true, only connect same polarity events

// Add these variables for graph rendering
GLuint g_graphVAO = 0;
GLuint g_graphVBO = 0;
GLuint g_graphColorVBO = 0;
size_t g_numGraphLines = 0;
std::vector<glm::vec3> g_graphVertices;
std::vector<glm::vec4> g_graphColors;

// Add these global variables - place them near other global variables related to video
FILE* g_ffmpegPipe = nullptr;
std::string g_outputVideoPath;
bool g_showRenderCompletePopup = false;  // Flag to show render complete popup

// Add near the other global variables related to rendering
float g_totalTasksDuration = 0.0f;
float g_elapsedTasksTime = 0.0f;

// Add this new function to scan for HDF5 files
void updateHDF5FileList() {
    g_hdf5Files.clear();
    namespace fs = std::filesystem;
    
    if (!fs::exists(g_dataDir)) {
        fs::create_directories(g_dataDir);
        return;
    }

    for (const auto& entry : fs::directory_iterator(g_dataDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            if (extension == ".h5" || extension == ".hdf5") {
                g_hdf5Files.push_back(entry.path().filename().string());
            }
        }
    }
    std::sort(g_hdf5Files.begin(), g_hdf5Files.end());
}

// ---------------- Function Definitions ---------------- //

// Compile shaders and return program ID.
GLuint createShaderProgram() {
    GLint success;
    char infoLog[512];
    
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex Shader Error:\n" << infoLog << std::endl;
    }
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment Shader Error:\n" << infoLog << std::endl;
    }
    
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader Program Linking Error:\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}



// Helper function to stop the loader thread safely.
void stopLoaderThread() {
    // Signal the loader thread to stop.
    g_stopLoading.store(true, std::memory_order_release);
    if (g_loaderThread.joinable()) {
        g_loaderThread.join();
    }
    // Reset the flag so that a new loader can be started.
    g_stopLoading.store(false, std::memory_order_release);
}
// Continuous Loader Thread Function
void continuousLoadEventsFromHDF5(const std::string& filename) {
    // Remove static so that local buffers are reused safely between iterations.
    std::vector<float> xs(FIXED_BUFFER_SIZE);
    std::vector<float> ys(FIXED_BUFFER_SIZE);
    std::vector<float> ts(FIXED_BUFFER_SIZE);
    std::vector<float> ps(FIXED_BUFFER_SIZE);

    bool simTimeAdjusted = false;
    size_t eventsToLoad = 0;

    while (!g_stopLoading.load()) {
        {
            // Acquire the global HDF5 mutex to serialize HDF5 calls.
            std::lock_guard<std::mutex> hdf5Lock(g_hdf5Mutex);

            // Initialize all HDF5 handles to an invalid value.
            hid_t file = -1;
            hid_t dataset_t = -1, dataset_x = -1, dataset_y = -1, dataset_p = -1;
            hid_t memspace = -1;
            
            // Open the file.
            file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (file < 0)
                break;
            
            // Open the "/CD/events/t" dataset.
            dataset_t = H5Dopen2(file, "/CD/events/t", H5P_DEFAULT);
            if (dataset_t < 0) {
                H5Fclose(file);
                file = -1;
                break;
            }
            
            // Immediately check for stop.
            if (g_stopLoading.load()) {
                H5Dclose(dataset_t);
                dataset_t = -1;
                H5Fclose(file);
                file = -1;
                break;
            }
            
            // Read total events from the timestamp dataset.
            hid_t space_t = H5Dget_space(dataset_t);
            hsize_t dims[1];
            H5Sget_simple_extent_dims(space_t, dims, NULL);
            size_t totalEvents = dims[0];
            H5Sclose(space_t);
            
            // Lambda to read a single timestamp.
            auto readTimestampAtIndex = [&](hsize_t index) -> float {
                hsize_t count = 1;
                hid_t mem_sp = H5Screate_simple(1, &count, NULL);
                hsize_t offset = index;
                hsize_t one = 1;
                hid_t file_sp = H5Dget_space(dataset_t);
                H5Sselect_hyperslab(file_sp, H5S_SELECT_SET, &offset, NULL, &one, NULL);
                float value;
                H5Dread(dataset_t, H5T_NATIVE_FLOAT, mem_sp, file_sp, H5P_DEFAULT, &value);
                H5Sclose(mem_sp);
                H5Sclose(file_sp);
                return value;
            };

            float windowStart = g_simTimeMicroSeconds;
            hsize_t startLow = 0, startHigh = totalEvents;
            while (startLow < startHigh) {
                hsize_t mid = (startLow + startHigh) / 2;
                float tsValue = readTimestampAtIndex(mid);
                if (tsValue < windowStart)
                    startLow = mid + 1;
                else
                    startHigh = mid;
                
                if (g_stopLoading.load()) {
                    H5Dclose(dataset_t);
                    dataset_t = -1;
                    H5Fclose(file);
                    file = -1;
                    return;
                }
            }
            hsize_t lowerIndex = startLow;
            eventsToLoad = std::min(FIXED_BUFFER_SIZE, static_cast<size_t>(totalEvents - lowerIndex));
            
            if (!simTimeAdjusted && eventsToLoad > 0) {
                float firstEventTime = readTimestampAtIndex(lowerIndex);
                g_simTimeMicroSeconds = firstEventTime;
                simTimeAdjusted = true;
                windowStart = g_simTimeMicroSeconds;
            }
            
            // Open the other datasets.
            dataset_x = H5Dopen2(file, "/CD/events/x", H5P_DEFAULT);
            dataset_y = H5Dopen2(file, "/CD/events/y", H5P_DEFAULT);
            dataset_p = H5Dopen2(file, "/CD/events/p", H5P_DEFAULT);
            if (dataset_x < 0 || dataset_y < 0 || dataset_p < 0) {
                if (dataset_x >= 0) { H5Dclose(dataset_x); dataset_x = -1; }
                if (dataset_y >= 0) { H5Dclose(dataset_y); dataset_y = -1; }
                if (dataset_p >= 0) { H5Dclose(dataset_p); dataset_p = -1; }
                H5Dclose(dataset_t);
                dataset_t = -1;
                H5Fclose(file);
                file = -1;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // Create memory space for bulk read.
            hsize_t count = eventsToLoad;
            memspace = H5Screate_simple(1, &count, NULL);

            // Read dataset X.
            {
                hsize_t offset = lowerIndex;
                hid_t filespace_x = H5Dget_space(dataset_x);
                H5Sselect_hyperslab(filespace_x, H5S_SELECT_SET, &offset, NULL, &count, NULL);
                H5Dread(dataset_x, H5T_NATIVE_FLOAT, memspace, filespace_x, H5P_DEFAULT, xs.data());
                H5Sclose(filespace_x);
            }
            
            // Read dataset Y.
            {
                hsize_t offset = lowerIndex;
                hid_t filespace_y = H5Dget_space(dataset_y);
                H5Sselect_hyperslab(filespace_y, H5S_SELECT_SET, &offset, NULL, &count, NULL);
                H5Dread(dataset_y, H5T_NATIVE_FLOAT, memspace, filespace_y, H5P_DEFAULT, ys.data());
                H5Sclose(filespace_y);
            }
            
            // Read dataset T.
            {
                hsize_t offset = lowerIndex;
                hid_t filespace_t = H5Dget_space(dataset_t);
                H5Sselect_hyperslab(filespace_t, H5S_SELECT_SET, &offset, NULL, &count, NULL);
                H5Dread(dataset_t, H5T_NATIVE_FLOAT, memspace, filespace_t, H5P_DEFAULT, ts.data());
                H5Sclose(filespace_t);
            }
            
            // Read dataset P.
            {
                hsize_t offset = lowerIndex;
                hid_t filespace_p = H5Dget_space(dataset_p);
                H5Sselect_hyperslab(filespace_p, H5S_SELECT_SET, &offset, NULL, &count, NULL);
                H5Dread(dataset_p, H5T_NATIVE_FLOAT, memspace, filespace_p, H5P_DEFAULT, ps.data());
                H5Sclose(filespace_p);
            }
            
            H5Sclose(memspace);
            memspace = -1;
            
            // Final stop check before processing.
            if (g_stopLoading.load()) {
                if (dataset_x >= 0) { H5Dclose(dataset_x); dataset_x = -1; }
                if (dataset_y >= 0) { H5Dclose(dataset_y); dataset_y = -1; }
                if (dataset_p >= 0) { H5Dclose(dataset_p); dataset_p = -1; }
                H5Dclose(dataset_t);
                dataset_t = -1;
                H5Fclose(file);
                file = -1;
                break;
            }
            
            // Cleanup: close every opened HDF5 object exactly once.
            if (dataset_x >= 0) { H5Dclose(dataset_x); dataset_x = -1; }
            if (dataset_y >= 0) { H5Dclose(dataset_y); dataset_y = -1; }
            if (dataset_p >= 0) { H5Dclose(dataset_p); dataset_p = -1; }
            if (dataset_t >= 0) { H5Dclose(dataset_t); dataset_t = -1; }
            if (file >= 0) { H5Fclose(file); file = -1; }
        } // End of HDF5 operations protected by g_hdf5Mutex

        // Build the events vector outside of the HDF5 lock.
        std::vector<Event> newEvents(eventsToLoad);
        for (size_t i = 0; i < eventsToLoad; ++i)
            newEvents[i] = { xs[i], ys[i], ts[i], ps[i] };

        {
            // Lock the events mutex when updating shared data.
            std::lock_guard<std::mutex> lock(g_eventsMutex);
            g_prefetchedEvents = std::move(newEvents);
            g_eventsLoaded = !g_prefetchedEvents.empty();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Cursor Position Callback
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    if (!g_isDragging)
        return;
    
    double dx = xpos - g_lastX;
    double dy = ypos - g_lastY;
    
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        dx *= TRANSLATION_SENSITIVITY;
        dy *= TRANSLATION_SENSITIVITY;
        g_camera.center += glm::vec3(-dx, dy, 0.0f);
    } else {
        g_camera.azimuth += static_cast<float>(dx) * ROTATION_SENSITIVITY;
        g_camera.elevation -= static_cast<float>(dy) * ROTATION_SENSITIVITY;
        if (g_camera.elevation > MAX_ELEVATION)
            g_camera.elevation = MAX_ELEVATION;
        if (g_camera.elevation < -MAX_ELEVATION)
            g_camera.elevation = -MAX_ELEVATION;
    }
    
    g_lastX = xpos;
    g_lastY = ypos;
}

// Mouse Button Callback
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_isDragging = true;
            glfwGetCursorPos(window, &g_lastX, &g_lastY);
        } else if (action == GLFW_RELEASE) {
            g_isDragging = false;
        }
    }
}

// Scroll Callback
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    
    g_camera.distance -= static_cast<float>(yoffset) * ZOOM_SENSITIVITY;
    if (g_camera.distance < MIN_CAMERA_DISTANCE)
        g_camera.distance = MIN_CAMERA_DISTANCE;
}

// Updated Keyboard Callback (keyCallback)
// Now, in addition to toggling pause (P), arrow keys adjust the camera center,
// and A, S, D adjust elevation, azimuth and distance respectively.
// Holding Shift with A, S, or D will reverse the direction of the change.
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    // If ImGui wants to capture the keyboard, let it do so.
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            // Adjust camera center (x and y) with arrow keys.
            case GLFW_KEY_LEFT:
                g_camera.center.x -= 10.0f;
                break;
            case GLFW_KEY_RIGHT:
                g_camera.center.x += 10.0f;
                break;
            case GLFW_KEY_UP:
                g_camera.center.y += 10.0f;
                break;
            case GLFW_KEY_DOWN:
                g_camera.center.y -= 10.0f;
                break;
            // Adjust elevation with A key (Shift for negative adjustment).
            case GLFW_KEY_A: {
                float step = 2.0f;
                if (mods & GLFW_MOD_SHIFT)
                    step = -step;
                g_camera.elevation += step;
                // Clamp elevation.
                if(g_camera.elevation > MAX_ELEVATION)
                    g_camera.elevation = MAX_ELEVATION;
                if(g_camera.elevation < -MAX_ELEVATION)
                    g_camera.elevation = -MAX_ELEVATION;
                break;
            }
            // Adjust azimuth with S key (Shift for negative adjustment).
            case GLFW_KEY_S: {
                float step = 2.0f;
                if (mods & GLFW_MOD_SHIFT)
                    step = -step;
                g_camera.azimuth += step;
                break;
            }
            // Adjust distance with D key (Shift for negative adjustment).
            case GLFW_KEY_D: {
                float step = 10.0f;
                if (mods & GLFW_MOD_SHIFT)
                    step = -step;
                g_camera.distance += step;
                if (g_camera.distance < MIN_CAMERA_DISTANCE)
                    g_camera.distance = MIN_CAMERA_DISTANCE;
                break;
            }
            default:
                break;
        }
    }

    // Toggle pause with the P key.
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
        g_paused = !g_paused;
        std::cout << (g_paused ? "Paused" : "Resumed") << std::endl;
    }
}

// Filter events for visualization within the time window.
std::vector<Event> filterEventsForVisualization(float simTime, float timeWindow) {
    std::vector<Event> visibleEvents;
    float windowStart = simTime;
    float windowEnd   = simTime + (timeWindow * 1e6);
    
    std::lock_guard<std::mutex> lock(g_eventsMutex);
    
    // Use binary search to locate the first event with t >= windowStart
    auto lowerIt = std::lower_bound(
        g_prefetchedEvents.begin(),
        g_prefetchedEvents.end(),
        windowStart,
        [](const Event& event, float t) {
            return event.t < t;
        }
    );
    
    // Iterate forward until events fall outside the visible time window.
    for (auto it = lowerIt; it != g_prefetchedEvents.end(); ++it) {
        if (it->t > windowEnd)
            break;
        visibleEvents.push_back(*it);
    }
    
    return visibleEvents;
}


// New function to build the event graph
void buildEventGraph(const std::vector<Event>& events) {
    // Clear previous graph data
    g_graphVertices.clear();
    g_graphColors.clear();
    
    // If there are too few events or graph visualization is disabled, return
    if (events.size() < 2 || !g_showEventGraph) {
        g_numGraphLines = 0;
        return;
    }
    
    const size_t maxEvents = std::min(events.size(), static_cast<size_t>(g_maxDisplayEvents));
    
    // Build a spatial grid for faster neighbor lookups
    // This is much more efficient than checking all pairs of events
    const float cellSize = g_graphMaxDistance;
    std::unordered_map<size_t, std::vector<size_t>> spatialGrid;
    
    auto hashPos = [cellSize](const glm::vec3& pos) -> size_t {
        // Simple spatial hashing function
        int x = static_cast<int>(pos.x / cellSize);
        int y = static_cast<int>(pos.y / cellSize);
        int z = static_cast<int>(pos.z / cellSize);
        return static_cast<size_t>((x * 73856093) ^ (y * 19349663) ^ (z * 83492791));
    };
    
    std::vector<glm::vec3> positions(maxEvents);
    std::vector<bool> isPositive(maxEvents);
    
    // Fill the spatial grid and position array
    for (size_t i = 0; i < maxEvents; i++) {
        const Event& e = events[i];
        
        // Create position vector based on current layout setting
        if (g_isVerticalLayout) {
            positions[i] = glm::vec3(e.y, e.x, 
                ((e.t - g_simTimeMicroSeconds) / (g_timeWindow * 1e6)) * g_windowSize - g_windowSize / 2.0f);
        } else {
            positions[i] = glm::vec3(e.x, e.y, 
                ((e.t - g_simTimeMicroSeconds) / (g_timeWindow * 1e6)) * g_windowSize - g_windowSize / 2.0f);
        }
        
        isPositive[i] = (e.p > 0.5f);
        
        // Skip negative events if we're only connecting positive
        if (g_connectPositiveOnly && !isPositive[i]) 
            continue;
        
        // Add to spatial grid
        size_t hash = hashPos(positions[i]);
        spatialGrid[hash].push_back(i);
    }
    
    // Reserve memory for graph (rough estimate)
    const size_t estEdgesPerEvent = std::min(g_graphMaxConnections, 10);
    g_graphVertices.reserve(maxEvents * estEdgesPerEvent * 2);
    g_graphColors.reserve(maxEvents * estEdgesPerEvent * 2);
    
    // For each event, find potential neighbors
    std::vector<int> connectionCounts(maxEvents, 0);
    for (size_t i = 0; i < maxEvents; i++) {
        // Skip if we have enough connections already
        if (connectionCounts[i] >= g_graphMaxConnections)
            continue;
            
        // Skip negative events if we're only connecting positive
        if (g_connectPositiveOnly && !isPositive[i]) 
            continue;
        
        // Check nearby grid cells
        const glm::vec3& pos = positions[i];
        size_t centerHash = hashPos(pos);
        
        // Check cells in a 3x3x3 neighborhood
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    glm::vec3 neighborCell = pos + glm::vec3(dx * cellSize, dy * cellSize, dz * cellSize);
                    size_t neighborHash = hashPos(neighborCell);
                    
                    auto it = spatialGrid.find(neighborHash);
                    if (it == spatialGrid.end())
                        continue;
                    
                    // Check events in this cell
                    for (size_t j : it->second) {
                        // Skip self connections and already full events
                        if (i == j || connectionCounts[j] >= g_graphMaxConnections)
                            continue;
                            
                        // Skip if polarity check fails
                        if (g_connectSamePolarity && isPositive[i] != isPositive[j])
                            continue;
                            
                        // Check actual distance
                        float dist = glm::distance(pos, positions[j]);
                        if (dist <= g_graphMaxDistance) {
                            // Apply probability filter
                            if (g_graphConnectionProb < 1.0f && static_cast<float>(rand()) / RAND_MAX > g_graphConnectionProb)
                                continue;
                                
                            // Add connection
                            g_graphVertices.push_back(pos);
                            g_graphVertices.push_back(positions[j]);
                            
                            // Use a color based on the event polarities
                            glm::vec4 lineColor = g_graphLineColor;
                            if (isPositive[i] && isPositive[j]) {
                                // Positive to positive: use base color
                            } else if (!isPositive[i] && !isPositive[j]) {
                                // Negative to negative: use a variation
                                lineColor.r *= 0.8f;
                                lineColor.g *= 0.8f;
                            } else {
                                // Mixed: another variation
                                lineColor.b *= 0.8f;
                            }
                            
                            // Fade color based on distance
                            float fadeMultiplier = 1.0f - (dist / g_graphMaxDistance);
                            lineColor.a *= fadeMultiplier * fadeMultiplier;  // Square for more pronounced fade
                            
                            g_graphColors.push_back(lineColor);
                            g_graphColors.push_back(lineColor);
                            
                            connectionCounts[i]++;
                            connectionCounts[j]++;
                            
                            // Stop if we've reached max connections
                            if (connectionCounts[i] >= g_graphMaxConnections)
                                break;
                        }
                    }
                    
                    if (connectionCounts[i] >= g_graphMaxConnections)
                        break;
                }
                if (connectionCounts[i] >= g_graphMaxConnections)
                    break;
            }
            if (connectionCounts[i] >= g_graphMaxConnections)
                break;
        }
    }
    
    g_numGraphLines = g_graphVertices.size() / 2;
}

// Add a function to draw the event graph
void drawEventGraph(GLuint shaderProgram, const glm::mat4& MVP) {
    if (!g_showEventGraph || g_numGraphLines == 0)
        return;
        
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    
    glBindVertexArray(g_graphVAO);
    glLineWidth(g_graphLineWidth);
    glDrawArrays(GL_LINES, 0, g_graphVertices.size());
    glLineWidth(1.0f);
    glBindVertexArray(0);
}


// Update the vertex buffers with events to render.
void updateEventBuffer(GLuint vbo, GLuint cbo) {
    if (!g_eventsLoaded)
        return;
    
    std::vector<glm::vec3> positions;
    std::vector<glm::vec4> colors;
    
    float windowStart = g_simTimeMicroSeconds;
    float windowEnd   = g_simTimeMicroSeconds + (g_timeWindow * 1e6);
    
    std::vector<Event> visualEvents = filterEventsForVisualization(g_simTimeMicroSeconds, g_timeWindow);
    
    if (visualEvents.size() > static_cast<size_t>(g_maxDisplayEvents)) {
        std::vector<Event> sampledEvents;
        sampledEvents.reserve(g_maxDisplayEvents);
        double step = static_cast<double>(visualEvents.size() - 1) / (g_maxDisplayEvents - 1);
        for (int i = 0; i < g_maxDisplayEvents; i++) {
            size_t idx = static_cast<size_t>(std::round(i * step));
            if (idx >= visualEvents.size())
                idx = visualEvents.size() - 1;
            sampledEvents.push_back(visualEvents[idx]);
        }
        visualEvents = std::move(sampledEvents);
    }
    
    for (const Event& event : visualEvents) {
        float normalizedTime = (event.t - windowStart) / (g_timeWindow * 1e6);
        // Map normalizedTime to z such that events span [-g_windowSize/2, g_windowSize/2]
        float zCoord = normalizedTime * g_windowSize - g_windowSize / 2.0f;
        if(g_isVerticalLayout) {
            positions.push_back(glm::vec3(event.y, event.x, zCoord));
        } else {
            positions.push_back(glm::vec3(event.x, event.y, zCoord));
        }
        colors.push_back(event.p > 0.5f ? g_positiveEventColor : g_negativeEventColor);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3),
                 positions.data(), GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(glm::vec4),
                 colors.data(), GL_DYNAMIC_DRAW);
    
    g_numPoints = positions.size();
    
    // Build the event graph if enabled
    if (g_showEventGraph) {
        buildEventGraph(visualEvents);
        
        // Update the graph buffers
        if (g_graphVAO == 0) {
            // First-time initialization
            glGenVertexArrays(1, &g_graphVAO);
            glBindVertexArray(g_graphVAO);
            
            glGenBuffers(1, &g_graphVBO);
            glBindBuffer(GL_ARRAY_BUFFER, g_graphVBO);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(0);
            
            glGenBuffers(1, &g_graphColorVBO);
            glBindBuffer(GL_ARRAY_BUFFER, g_graphColorVBO);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(1);
            
            glBindVertexArray(0);
        }
        
        // Update the buffer data
        glBindBuffer(GL_ARRAY_BUFFER, g_graphVBO);
        glBufferData(GL_ARRAY_BUFFER, g_graphVertices.size() * sizeof(glm::vec3),
                     g_graphVertices.data(), GL_DYNAMIC_DRAW);
        
        glBindBuffer(GL_ARRAY_BUFFER, g_graphColorVBO);
        glBufferData(GL_ARRAY_BUFFER, g_graphColors.size() * sizeof(glm::vec4),
                     g_graphColors.data(), GL_DYNAMIC_DRAW);
    }
}

// Setup coordinate axes buffers.
void setupCoordinateAxes() {
    std::vector<glm::vec3> axisVertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(AXIS_LENGTH, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, AXIS_LENGTH, 0.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, AXIS_LENGTH)
    };

    std::vector<glm::vec4> axisColors = {
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)
    };

    glGenVertexArrays(1, &g_axisVAO);
    glBindVertexArray(g_axisVAO);

    glGenBuffers(1, &g_axisVBO);
    glBindBuffer(GL_ARRAY_BUFFER, g_axisVBO);
    glBufferData(GL_ARRAY_BUFFER, axisVertices.size() * sizeof(glm::vec3), 
                 axisVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &g_axisColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, g_axisColorVBO);
    glBufferData(GL_ARRAY_BUFFER, axisColors.size() * sizeof(glm::vec4), 
                 axisColors.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// Draw coordinate axes.
void drawCoordinateAxes(GLuint shaderProgram, const glm::mat4& MVP) {
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    
    glBindVertexArray(g_axisVAO);
    glLineWidth(AXIS_LINE_WIDTH);
    glDrawArrays(GL_LINES, 0, 6);
    glLineWidth(1.0f);
    glBindVertexArray(0);
}

// Execute a shell command and return its output.
std::string execCommand(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
        return "Command failed";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
        result += buffer.data();
    return result;
}

// Get CPU usage.
std::string getCPUUsage() {
#ifdef __APPLE__
    return execCommand("top -l 1 | grep -E '^CPU' | awk '{print $3}'");
#else
    return execCommand("top -bn1 | grep 'Cpu(s)' | awk '{print $2}'");
#endif
}

// Get RAM usage.
std::string getRAMUsage() {
#ifdef __APPLE__
    return execCommand("top -l 1 | grep -E '^PhysMem' | awk '{print $2}'");
#else
    return execCommand("free -m | grep Mem | awk '{print $3/$2 * 100}'");
#endif
}

// Get GPU usage.
std::string getGPUUsage() {
#ifdef __APPLE__
    return "N/A";
#else
    return execCommand("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader");
#endif
}

// System monitoring thread.
void systemMonitoringThread() {
    while (!g_shouldStopMonitoring) {
        std::string newCpuUsage = getCPUUsage();
        std::string newRamUsage = getRAMUsage();
        std::string newGpuUsage = getGPUUsage();

        {
            std::lock_guard<std::mutex> lock(g_statsMutex);
            g_systemStats.cpuUsage = newCpuUsage;
            g_systemStats.ramUsage = newRamUsage;
            g_systemStats.gpuUsage = newGpuUsage;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// Update FPS estimate.
void updateFPS(double deltaTimeSeconds, double& avgFps) {
    g_fpsHistory[g_fpsIndex] = 1.0 / deltaTimeSeconds;
    g_fpsIndex = (g_fpsIndex + 1) % FPS_SAMPLE_COUNT;
    
    avgFps = 0.0;
    for (double fps : g_fpsHistory) {
        avgFps += fps;
    }
    avgFps /= FPS_SAMPLE_COUNT;
}

// Update simulation state.
void updateSimulation(double deltaTimeSeconds, GLuint vbo, GLuint cbo) {
    if (!g_paused) {
        // Smooth the slomo factor changes to avoid abrupt simulation time jumps (otherwise, the simulation time jumps which leads to events being skipped).
        const float smoothingRate = 2.0f;  // You can adjust this rate to achieve more or less smoothing.
        float dt = static_cast<float>(deltaTimeSeconds);
        g_slomoFactorSmooth += (g_slomoFactor - g_slomoFactorSmooth) * std::min(dt * smoothingRate, 1.0f);
        
        // Update simulation time using the smoothed slomo factor.
        g_simTimeMicroSeconds += (deltaTimeSeconds * g_slomoFactorSmooth) * 1e6;

        // std::cout << "End time reached. Times (microseconds):" << std::endl;
        // std::cout << "  Start time: " << g_startTimeMicroSeconds << std::endl;
        // std::cout << "  Current time: " << g_simTimeMicroSeconds << std::endl; 
        // std::cout << "  End time: " << g_endTimeMicroSeconds << std::endl;
        // If the simulation time reaches (or exceeds) the end, loop back to zero.
        if (g_endTimeMicroSeconds > 0.0f && g_simTimeMicroSeconds >= g_endTimeMicroSeconds) {
            g_simTimeMicroSeconds = g_startTimeMicroSeconds;
        }
            
        updateEventBuffer(vbo, cbo);
    }
    
}

// Reset the camera to its default position.
void resetCamera() {
    g_camera.center  = glm::vec3(EVENT_VOLUME_WIDTH / 2.0f, EVENT_VOLUME_HEIGHT / 2.0f, 0);
    g_camera.azimuth = 0.0f;
    g_camera.elevation = 0.0f;
    g_camera.distance = 2200.0f;
}

// Save the task list in JSON format into a dedicated folder "task_lists".
void saveTaskListToJson(const std::string &taskListName) {
    namespace fs = std::filesystem;
    // Create the task_lists folder if it does not exist.
    fs::path taskFolder("task_lists");
    if (!fs::exists(taskFolder)) {
        fs::create_directories(taskFolder);
    }
    std::string filename = taskListName;
    // Ensure the filename ends with .json
    if (filename.find(".json") == std::string::npos)
        filename += ".json";
    fs::path filePath = taskFolder / filename;

    // Build a JSON object that contains the task list name and tasks.
    json j;
    j["task_list_name"] = taskListName;
    // Add metadata for data directory and HDF5 file
    j["data_dir"] = g_dataDir;
    j["hdf5_file"] = g_selectedFileIndex >= 0 ? g_hdf5Files[g_selectedFileIndex] : "";
    
    // Save GUI settings
    j["gui_settings"] = {
        {"max_display_events", g_maxDisplayEvents},
        {"time_window", g_timeWindow},
        {"point_size", g_pointSize},
        {"window_size", g_windowSize},
        {"volume_rotation_index", g_volumeRotationIndex},
        {"slomo_factor", g_slomoFactor},
        {"show_volume_box", g_showVolumeBox},
        {"show_coordinate_axes", g_showCoordinateAxes},
        {"background_color", {g_backgroundColor.r, g_backgroundColor.g, g_backgroundColor.b, g_backgroundColor.a}},
        {"positive_event_color", {g_positiveEventColor.r, g_positiveEventColor.g, g_positiveEventColor.b, g_positiveEventColor.a}},
        {"negative_event_color", {g_negativeEventColor.r, g_negativeEventColor.g, g_negativeEventColor.b, g_negativeEventColor.a}}
    };

    j["tasks"] = json::array();
    for (const auto &task : g_cameraTasks) {
        json jTask;
        // Save the task type as a string.
        switch (task.type) {
            case TaskType::CameraMove:
                jTask["type"] = "CameraMove";
                break;
            case TaskType::SetSimTime:
                jTask["type"] = "SetSimTime";
                break;
            case TaskType::Wait:
                jTask["type"] = "Wait";
                break;
            case TaskType::SetZLength:
                jTask["type"] = "SetZLength";
                jTask["targetWindowSize"] = task.targetWindowSize;
                break;
            case TaskType::CameraRotate:
                jTask["type"] = "CameraRotate";
                jTask["rotationAngle"] = task.rotationAngle;
                jTask["rotationAxis"] = { task.rotationAxis.x, task.rotationAxis.y, task.rotationAxis.z };
                break;
            case TaskType::SetSlomoFactor:
                jTask["type"] = "SetSlomoFactor";
                jTask["targetSlomoFactor"] = task.targetSlomoFactor;
                break;
        }
        jTask["name"] = task.name;
        jTask["duration"] = task.duration;
        jTask["simTime"] = task.simTime;

        json jCamera;
        jCamera["center"] = { task.camera.center.x, task.camera.center.y, task.camera.center.z };
        jCamera["elevation"] = task.camera.elevation;
        jCamera["azimuth"] = task.camera.azimuth;
        jCamera["distance"] = task.camera.distance;
        jTask["camera"] = jCamera;

        j["tasks"].push_back(jTask);
    }

    std::ofstream out(filePath.string());
    if (!out.is_open()) {
        std::cerr << "Error writing task list to file: " << filePath << "\n";
        return;
    }
    out << j.dump(4);
    out.close();
    std::cout << "Task list saved to " << filePath << std::endl;
}


bool setStartEndTime(const std::string &filename) {
    namespace fs = std::filesystem;
    if (!fs::exists(filename)) {
        std::cerr << "setStartEndTime: file does not exist: " << filename << std::endl;
        g_startTimeMicroSeconds = 0.0f;
        g_endTimeMicroSeconds = 0.0f;
        return false;
    }

    // Ensure no other thread is making HDF5 calls concurrently.
    std::lock_guard<std::mutex> lock(g_hdf5Mutex);

    // Open the HDF5 file.
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        g_startTimeMicroSeconds = 0.0f;
        g_endTimeMicroSeconds = 0.0f;
        return false;
    }

    // Open the dataset.
    hid_t dataset = H5Dopen2(file, "/CD/events/t", H5P_DEFAULT);
    if (dataset < 0) {
        std::cerr << "Failed to open dataset /CD/events/t in file: " << filename << std::endl;
        H5Fclose(file);
        g_startTimeMicroSeconds = 0.0f;
        g_endTimeMicroSeconds = 0.0f;
        return false;
    }

    // Get the dataspace.
    hid_t space = H5Dget_space(dataset);
    if (space < 0) {
        std::cerr << "Failed to get dataspace from /CD/events/t" << std::endl;
        H5Dclose(dataset);
        H5Fclose(file);
        g_startTimeMicroSeconds = 0.0f;
        g_endTimeMicroSeconds = 0.0f;
        return false;
    }

    // Get the total number of events.
    hsize_t dims[1] = {0};
    if (H5Sget_simple_extent_dims(space, dims, NULL) < 0) {
        std::cerr << "Failed to get dataset dimensions" << std::endl;
        H5Sclose(space);
        H5Dclose(dataset);
        H5Fclose(file);
        g_startTimeMicroSeconds = 0.0f;
        g_endTimeMicroSeconds = 0.0f;
        return false;
    }
    size_t totalEvents = dims[0];
    H5Sclose(space);

    if (totalEvents > 0) {
        float firstTimestamp = 0.0f, lastTimestamp = 0.0f;
        hsize_t count = 1;

        // Read first timestamp.
        hsize_t offset = 0;
        hid_t memspace = H5Screate_simple(1, &count, NULL);
        if(memspace < 0) {
            std::cerr << "Failed to create memory space for first timestamp" << std::endl;
            H5Dclose(dataset);
            H5Fclose(file);
            return false;
        }
        hid_t filespace = H5Dget_space(dataset);
        if(filespace < 0) {
            std::cerr << "Failed to get filespace for first timestamp" << std::endl;
            H5Sclose(memspace);
            H5Dclose(dataset);
            H5Fclose(file);
            return false;
        }
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL) < 0) {
            std::cerr << "Failed to select hyperslab for first timestamp" << std::endl;
        }
        H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, &firstTimestamp);
        H5Sclose(filespace);
        H5Sclose(memspace);

        // Read last timestamp.
        offset = totalEvents - 1;
        memspace = H5Screate_simple(1, &count, NULL);
        if(memspace < 0) {
            std::cerr << "Failed to create memory space for last timestamp" << std::endl;
            H5Dclose(dataset);
            H5Fclose(file);
            return false;
        }
        filespace = H5Dget_space(dataset);
        if(filespace < 0) {
            std::cerr << "Failed to get filespace for last timestamp" << std::endl;
            H5Sclose(memspace);
            H5Dclose(dataset);
            H5Fclose(file);
            return false;
        }
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &count, NULL) < 0) {
            std::cerr << "Failed to select hyperslab for last timestamp" << std::endl;
        }
        H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, &lastTimestamp);
        H5Sclose(filespace);
        H5Sclose(memspace);

        g_startTimeMicroSeconds = firstTimestamp;
        g_endTimeMicroSeconds = lastTimestamp;
    } else {
        g_startTimeMicroSeconds = 0.0f;
        g_endTimeMicroSeconds = 0.0f;
    }

    H5Dclose(dataset);
    H5Fclose(file);
    return true;
}

// Switch to a new file, ensuring that the old file's HDF5 handle is closed 
// and that previous event data is completely cleared.
void switchFile(const std::string &newFilePath) {
    // STEP 1. Stop and join the existing loader thread.
    g_stopLoading.store(true, std::memory_order_release);
    if (g_loaderThread.joinable()) {
        g_loaderThread.join();
    }
    
    // STEP 2. Clear CPU event data.
    {
        std::lock_guard<std::mutex> lock(g_eventsMutex);
        g_prefetchedEvents.clear();
        g_eventsLoaded = false;
    }
    
    // Also reset any simulation counters.
    g_numPoints = 0;
    g_simTimeMicroSeconds = 0.0f;
    g_endTimeMicroSeconds = 0.0f;
    g_cameraTasks.clear();
    
    // STEP 3. Instruct the render thread to clear GPU buffers, only if initialized.
    if (g_buffersInitialized) {
        g_shouldClearBuffers = true;
    }
    
    // STEP 4. Update the global file path.
    g_eventFilePath = newFilePath;
    
    // STEP 5. Optionally update the simulation time boundaries from the new file.
    if (!g_eventFilePath.empty()) {
        setStartEndTime(g_eventFilePath);
    }
    
    // STEP 6. Restart the loader thread on the new file.
    g_stopLoading.store(false, std::memory_order_release);
    g_loaderThread = std::thread(continuousLoadEventsFromHDF5, g_eventFilePath);
}

void loadTaskListFromJson(const std::string &taskListName) {
    namespace fs = std::filesystem;
    std::string filename = taskListName;
    // Ensure the filename ends with .json
    if (filename.find(".json") == std::string::npos)
        filename += ".json";
    fs::path taskFolder("task_lists");
    fs::path filePath = taskFolder / filename;

    json j;
    std::ifstream in(filePath.string());
    if (!in) {
        std::cerr << "Error loading task list from file: " << filePath << "\n";
        return;
    }
    in >> j;
    in.close();

    // Update the global task list name from the file.
    std::string loadedTaskListName = j.value("task_list_name", taskListName);
    std::strncpy(g_taskListNameBuffer, loadedTaskListName.c_str(), sizeof(g_taskListNameBuffer));
    g_taskListNameBuffer[sizeof(g_taskListNameBuffer) - 1] = '\0';

    // Load and set the data directory and HDF5 file
    if (j.contains("data_dir")) {
        g_dataDir = j["data_dir"];
        updateHDF5FileList();
    }
    
    if (j.contains("hdf5_file")) {
        std::string hdf5File = j["hdf5_file"];
        // Find the index of the HDF5 file in the current list
        auto it = std::find(g_hdf5Files.begin(), g_hdf5Files.end(), hdf5File);
        if (it != g_hdf5Files.end()) {
            g_selectedFileIndex = static_cast<int>(std::distance<std::vector<std::string>::iterator>(g_hdf5Files.begin(), it));
            // Construct the full file path
            std::string newFilePath = (std::filesystem::path(g_dataDir) / hdf5File).string();
            // Use the common switchFile function for consistent behavior
            switchFile(newFilePath);
        }
    }

    g_cameraTasks.clear();
    try {
        for (const auto &jTask : j["tasks"]) {
            CameraTask task;
            std::string typeStr = jTask.value("type", "CameraMove");
            if (typeStr == "CameraMove") {
                task.type = TaskType::CameraMove;
            } else if (typeStr == "SetSimTime") {
                task.type = TaskType::SetSimTime;
            } else if (typeStr == "Wait") {
                task.type = TaskType::Wait;
            } else if (typeStr == "SetZLength") {
                task.type = TaskType::SetZLength;
                task.targetWindowSize = jTask.value("targetWindowSize", 1000.0f);
            } else if (typeStr == "CameraRotate") {
                task.type = TaskType::CameraRotate;
                task.rotationAngle = jTask.value("rotationAngle", 0.0f);
                if(jTask.contains("rotationAxis") && jTask["rotationAxis"].is_array() && jTask["rotationAxis"].size() == 3) {
                    task.rotationAxis = glm::vec3(jTask["rotationAxis"][0].get<float>(),
                                                  jTask["rotationAxis"][1].get<float>(),
                                                  jTask["rotationAxis"][2].get<float>());
                }
            } else if (typeStr == "SetSlomoFactor") {
                task.type = TaskType::SetSlomoFactor;
                task.targetSlomoFactor = jTask.value("targetSlomoFactor", INITIAL_SLOMO_FACTOR);
            }
            task.name = jTask.value("name", "");
            task.duration = jTask.value("duration", 3.0f);
            task.simTime = jTask.value("simTime", 0.0f);

            json jCamera = jTask["camera"];
            task.camera.center = glm::vec3(jCamera["center"][0].get<float>(),
                                           jCamera["center"][1].get<float>(),
                                           jCamera["center"][2].get<float>());
            task.camera.elevation = jCamera.value("elevation", 0.0f);
            task.camera.azimuth = jCamera.value("azimuth", 0.0f);
            task.camera.distance = jCamera.value("distance", 500.0f);
            g_cameraTasks.push_back(task);
        }
    }
    catch (std::exception &ex) {
        std::cerr << "Error parsing task file: " << ex.what() << std::endl;
    }
    std::cout << "Task list loaded from " << filePath << std::endl;

    // Load GUI settings
    if (j.contains("gui_settings")) {
        const auto& settings = j["gui_settings"];
        g_maxDisplayEvents = settings.value("max_display_events", g_maxDisplayEvents);
        g_timeWindow = settings.value("time_window", g_timeWindow);
        g_pointSize = settings.value("point_size", g_pointSize);
        g_windowSize = settings.value("window_size", g_windowSize);
        g_volumeRotationIndex = settings.value("volume_rotation_index", g_volumeRotationIndex);
        g_slomoFactor = settings.value("slomo_factor", g_slomoFactor);
        g_showVolumeBox = settings.value("show_volume_box", g_showVolumeBox);
        g_showCoordinateAxes = settings.value("show_coordinate_axes", g_showCoordinateAxes);
        
        if (settings.contains("background_color")) {
            const auto& bgColor = settings["background_color"];
            g_backgroundColor = glm::vec4(
                bgColor[0].get<float>(),
                bgColor[1].get<float>(),
                bgColor[2].get<float>(),
                bgColor[3].get<float>()
            );
        }
        
        if (settings.contains("positive_event_color")) {
            const auto& posColor = settings["positive_event_color"];
            g_positiveEventColor = glm::vec4(
                posColor[0].get<float>(),
                posColor[1].get<float>(),
                posColor[2].get<float>(),
                posColor[3].get<float>()
            );
        }
        
        if (settings.contains("negative_event_color")) {
            const auto& negColor = settings["negative_event_color"];
            g_negativeEventColor = glm::vec4(
                negColor[0].get<float>(),
                negColor[1].get<float>(),
                negColor[2].get<float>(),
                negColor[3].get<float>()
            );
        }
    }
}


float smootherstep(float t) {
    // Clamp t to [0,1]
    if(t < 0.f)
        t = 0.f;
    if(t > 1.f)
        t = 1.f;
    return t * t * t * (t * (6.0f * t - 15.0f) + 10.0f);
}




// Clears the CPU-side event buffer and also resets GPU buffers.
void clearEventBuffers(GLuint eventVBO, GLuint colorVBO) {
    {
        std::lock_guard<std::mutex> lock(g_eventsMutex);
        g_prefetchedEvents.clear();
        g_eventsLoaded = false;
    }
    
    // Also reset the number of rendered points.
    g_numPoints = 0;
    
    // Only attempt to clear GL buffers if they're initialized
    if (!g_buffersInitialized || eventVBO == 0 || colorVBO == 0) {
        std::cerr << "Warning: Attempted to clear invalid buffer(s)" << std::endl;
        return;
    }
    
    // Clear the GPU buffers that hold the event positions and colors.
    glBindBuffer(GL_ARRAY_BUFFER, eventVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
}

// Helper function to convert HSV to RGB.
glm::vec3 hsvToRgb(float h, float s, float v) {
    float C = s * v;
    float X = C * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = v - C;
    glm::vec3 rgb;
    if (h < 60.0f) {
        rgb = glm::vec3(C, X, 0.0f);
    } else if (h < 120.0f) {
        rgb = glm::vec3(X, C, 0.0f);
    } else if (h < 180.0f) {
        rgb = glm::vec3(0.0f, C, X);
    } else if (h < 240.0f) {
        rgb = glm::vec3(0.0f, X, C);
    } else if (h < 300.0f) {
        rgb = glm::vec3(X, 0.0f, C);
    } else {
        rgb = glm::vec3(C, 0.0f, X);
    }
    return rgb + glm::vec3(m);
}

// Add this helper function near the other utility functions
std::string getUniqueVideoFilename(const std::string& baseFilename) {
    namespace fs = std::filesystem;
    
    // Split the filename into base and extension
    size_t dotPos = baseFilename.find_last_of('.');
    std::string name = (dotPos != std::string::npos) ? baseFilename.substr(0, dotPos) : baseFilename;
    std::string ext = (dotPos != std::string::npos) ? baseFilename.substr(dotPos) : "";
    
    // Sanitize the name - replace problematic characters with underscores
    std::string sanitizedName = name;
    // Replace spaces, colons, and other problematic characters
    const std::string problematicChars = " :;,?*|\"<>";
    for (char c : problematicChars) {
        std::replace(sanitizedName.begin(), sanitizedName.end(), c, '_');
    }
    
    // Check if the sanitized filename exists in the render folder
    fs::path renderFolder("render");
    fs::path currentPath = renderFolder / (sanitizedName + ext);
    
    // If the sanitized filename doesn't exist, use it
    if (!fs::exists(currentPath)) {
        return sanitizedName + ext;
    }
    
    // If it exists, try adding numbers until we find an available name
    int counter = 2;
    std::string currentName;
    do {
        currentName = sanitizedName + "_" + std::to_string(counter) + ext;
        currentPath = renderFolder / currentName;
        counter++;
    } while (fs::exists(currentPath));
    
    std::cout << "File '" << sanitizedName + ext << "' already exists, using '" << currentName << "' instead." << std::endl;

    return currentName;
}

// Function: initializeFFmpegPipe
// This function sets up an ffmpeg pipe for streaming video frames directly to disk.
// It configures the output resolution, format, and file path but doesn't write any frames.
void initializeFFmpegPipe() {
    namespace fs = std::filesystem;
    // Ensure the render folder exists.
    fs::path renderFolder("render");
    if (!fs::exists(renderFolder)) {
        fs::create_directories(renderFolder);
    }
    
    // Get the base name of the event file (from CLI) without extension.
    std::string eventFileBase = fs::path(g_eventFilePath).stem().string();
    // Use the current task list name.
    std::string taskListName = std::string(g_taskListNameBuffer);
    // Build the output filename.
    std::string outputFilename = eventFileBase + "_" + taskListName + ".mp4";
    fs::path outputPath = renderFolder / outputFilename;
    
    // Get a unique filename to avoid overwriting existing files
    outputPath = renderFolder / getUniqueVideoFilename(outputPath.filename().string());
    g_outputVideoPath = outputPath.string();

    std::ostringstream ffmpegCmd;
    ffmpegCmd << "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size " 
              << g_videoWidth << "x" << g_videoHeight
              << " -framerate 60 -i - -vf vflip -c:v libx264 -pix_fmt yuv420p "
              << "-movflags frag_keyframe+empty_moov+faststart "
              << "-g 30 -keyint_min 30 "  // Force keyframes every 30 frames for better seeking and streaming
              << outputPath.string();
    
    g_ffmpegPipe = popen(ffmpegCmd.str().c_str(), "w");
    if (!g_ffmpegPipe) {
        std::cerr << "Failed to open ffmpeg pipe." << std::endl;
        g_videoSaving = false;
        g_renderMode = false;
    }
}

// Add a new function to write a single frame to the ffmpeg pipe
void writeFrameToFFmpeg(const std::vector<unsigned char>& frameBuffer) {
    if (!g_ffmpegPipe) return;
    
    size_t written = fwrite(frameBuffer.data(), 1, frameBuffer.size(), g_ffmpegPipe);
    if (written != frameBuffer.size()) {
        std::cerr << "Failed to write frame to ffmpeg pipe" << std::endl;
    }
    fflush(g_ffmpegPipe);
}

// Add a function to close the ffmpeg pipe
void closeFFmpegPipe() {
    if (g_ffmpegPipe) {
        pclose(g_ffmpegPipe);
        g_ffmpegPipe = nullptr;
        std::cout << "Video saved to " << g_outputVideoPath << std::endl;
    }
}

// Add this function before renderImGuiInterface to calculate the total rendering time
float calculateTotalTasksDuration() {
    float totalDuration = 0.0f;
    for (const auto& task : g_cameraTasks) {
        totalDuration += task.duration;
    }
    return totalDuration;
}

// Add this function to calculate the elapsed time based on completed tasks and current task progress
float calculateElapsedTasksTime() {
    if (g_cameraTasks.empty()) return 0.0f;
    
    float elapsedTime = 0.0f;
    
    // Add up the duration of all completed tasks
    for (size_t i = 0; i < g_currentTaskIndex; i++) {
        elapsedTime += g_cameraTasks[i].duration;
    }
    
    // Add the progress of the current task
    if (g_currentTaskIndex < g_cameraTasks.size()) {
        float currentTaskProgress = g_taskTimeAccumulator / g_cameraTasks[g_currentTaskIndex].duration;
        elapsedTime += currentTaskProgress * g_cameraTasks[g_currentTaskIndex].duration;
    }
    
    return elapsedTime;
}

// Render ImGui interface.
void renderImGuiInterface(double avgFps, int windowWidth, int windowHeight) {
    // Set the popup flag
    static bool displayPopup = false;
    if (g_showRenderCompletePopup) {
        displayPopup = true;
        g_showRenderCompletePopup = false;
    }
    
    float guiWidth = static_cast<float>(g_GUI_FIXED_WIDTH);
    ImGui::SetNextWindowPos(ImVec2(windowWidth - guiWidth, 0));
    ImGui::SetNextWindowSize(ImVec2(guiWidth, static_cast<float>(windowHeight)));
    ImGui::Begin("Controls", nullptr,
                   ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoCollapse);

    // File selection combo box
    if (ImGui::BeginCombo("HDF5 File", g_selectedFileIndex >= 0 ? g_hdf5Files[g_selectedFileIndex].c_str() : "Select File")) {
        for (int i = 0; i < g_hdf5Files.size(); i++) {
            bool isSelected = (g_selectedFileIndex == i);
            if (ImGui::Selectable(g_hdf5Files[i].c_str(), isSelected)) {
                // Only act if a different file is selected.
                if (g_selectedFileIndex != i) {
                    g_selectedFileIndex = i;
                    // Construct the new file path.
                    std::string newFilePath = (std::filesystem::path(g_dataDir) / g_hdf5Files[i]).string();
                    // Switch file: stop the current loader thread, clear the current data,
                    // update globals, and start a new loader thread for the new file.
                    switchFile(newFilePath);
                }
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Separator();

    if (ImGui::Button(g_paused ? "Continue" : "Pause"))
        g_paused = !g_paused;
    // Convert simulation time (stored in microseconds) into seconds.
    float simTimeSec = g_simTimeMicroSeconds / 1e6f;
    // The slider range is set using the min and max event times (converted to seconds).
    if (ImGui::SliderFloat("Sim Time (s)", &simTimeSec, g_startTimeMicroSeconds / 1e6f, g_endTimeMicroSeconds / 1e6f, "%.3f s")) {
        // When the slider is adjusted, update the simulation time (back in microseconds).
        g_simTimeMicroSeconds = simTimeSec * 1e6f;
    }
    
    ImGui::SliderFloat("Slomo Factor", &g_slomoFactor, MIN_SLOMO_FACTOR, MAX_SLOMO_FACTOR, "%.2fx");
    ImGui::SliderFloat("Z Length", &g_windowSize, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE);
    ImGui::SliderFloat("Time Window (s)", &g_timeWindow, 0.001f, 5.0f);
    ImGui::SliderFloat("Point Size", &g_pointSize, MIN_POINT_SIZE, MAX_POINT_SIZE);
    ImGui::SliderInt("Max Events", &g_maxDisplayEvents, 1000, 1000000);
    const char* rotationItems[] = { "0°", "90°", "180°", "270°" };
    ImGui::Combo("Volume Rotation", &g_volumeRotationIndex, rotationItems, IM_ARRAYSIZE(rotationItems));

    ImGui::Separator();
    ImGui::Text("Tasks:");

    // Row of task control buttons.
    if (ImGui::Button("Run Tasks") && g_cameraTasks.size() >= 1) {
        g_runningTasks = true;
        g_currentTaskIndex = 0;
        g_taskTimeAccumulator = 0.0f;
        g_startCamera = g_camera;  // Record the current camera pose for full-run mode.
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop Tasks", ImVec2(0, 0))) {
        // Stop all running tasks without clearing the task list.
        g_runningTasks = false;
        g_isRecording = false;
        g_individualTaskRun = false;
        g_taskTimeAccumulator = 0.0f;
        // Close ffmpeg pipe if it's open
        if (g_renderMode) {
            closeFFmpegPipe();
            g_videoSaving = false;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear Tasks", ImVec2(0,0))) {
        g_cameraTasks.clear();
        g_runningTasks = false;
    }
    
    ImGui::Separator();
    // Buttons for adding tasks.
    if (ImGui::Button("+ CameraMove")) {
        CameraTask task;
        task.type = TaskType::CameraMove;
        task.camera = g_camera;
        task.name = "CameraMove " + std::to_string(g_cameraTaskCounter++);
        task.duration = 2.0f;
        g_cameraTasks.push_back(task);
    }
    ImGui::SameLine();
    if (ImGui::Button("+ SetSimTime")) {
        CameraTask task;
        task.type = TaskType::SetSimTime;
        task.name = "SetSimTime " + std::to_string(g_cameraTaskCounter++);
        // Set target simulation time (in seconds).
        task.simTime = g_simTimeMicroSeconds / 1e6f;
        g_cameraTasks.push_back(task);
    }
    ImGui::SameLine();
    if (ImGui::Button("+ Wait")) {
        CameraTask task;
        task.type = TaskType::Wait;
        task.name = "Wait " + std::to_string(g_cameraTaskCounter++);
        task.duration = 1.0f;  // Default wait time (in seconds)
        g_cameraTasks.push_back(task);
    }
    
    if (ImGui::Button("+ SetZLength")) {  // ---- New button for setting Z Length
        CameraTask task;
        task.type = TaskType::SetZLength;
        task.name = "SetZLength " + std::to_string(g_cameraTaskCounter++);
        task.duration = 2.0f; // Default duration for the transition
        task.targetWindowSize = g_windowSize; // Use the current g_windowSize as the target
        task.startWindowSize = 0.0f; // (To be initialized when the task starts)
        g_cameraTasks.push_back(task);
    }
    ImGui::SameLine();
    if (ImGui::Button("+ CameraRotate")) { // ---- New button for CameraRotate
        CameraTask task;
        task.type = TaskType::CameraRotate;
        task.name = "CameraRotate " + std::to_string(g_cameraTaskCounter++);
        task.duration = 2.0f;    // Default duration
        g_cameraTasks.push_back(task);
    }
    ImGui::SameLine();
    if (ImGui::Button("+ SetSlomoFactor")) { // ---- New button for SetSlomoFactor
        CameraTask task;
        task.type = TaskType::SetSlomoFactor;
        task.name = "SetSlomoFactor " + std::to_string(g_cameraTaskCounter++);
        task.duration = 2.0f;    // Default duration
        task.targetSlomoFactor = g_slomoFactor; // Use the current g_slomoFactor as the target
        g_cameraTasks.push_back(task);
    }

    ImGui::Separator();
    ImGui::InputText("Task List Name", g_taskListNameBuffer, IM_ARRAYSIZE(g_taskListNameBuffer));
    ImGui::BeginChild("TaskListChild", ImVec2(0, 300), true);
    for (size_t i = 0; i < g_cameraTasks.size(); i++) {
        ImGui::PushID(i);
        // Display task name; highlight if currently running.
        if (g_runningTasks && (i == g_currentTaskIndex)) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f)); // green
            ImGui::Text("%s", g_cameraTasks[i].name.c_str());
            ImGui::PopStyleColor();
        } else {
            ImGui::Text("%s", g_cameraTasks[i].name.c_str());
        }
        
        ImGui::SameLine();
        if (ImGui::SmallButton("Run")) {
            if (g_cameraTasks[i].type == TaskType::CameraMove) {
                g_individualTrajectory.clear();
                g_individualTrajectory.push_back(g_camera);  // current camera pose as start
                g_individualTrajectory.push_back(g_cameraTasks[i].camera);  // target pose
                g_individualSegmentDuration = g_cameraTasks[i].duration;
                g_individualTaskType = TaskType::CameraMove;
                g_currentTaskIndex = i; // set to the clicked task's index
                g_taskTimeAccumulator = 0.0f;
                g_individualTaskRun = true;
                g_runningTasks = true;
            } else if (g_cameraTasks[i].type == TaskType::SetSimTime) {
                g_currentTaskIndex = i;
                g_simTimeMicroSeconds = g_cameraTasks[i].simTime * 1e6f;
                g_individualTaskRun = false;
                g_runningTasks = false;
                g_taskTimeAccumulator = 0.0f;
            } else if (g_cameraTasks[i].type == TaskType::Wait) {
                g_individualSegmentDuration = g_cameraTasks[i].duration;
                g_individualTaskType = TaskType::Wait;
                g_currentTaskIndex = i;
                g_taskTimeAccumulator = 0.0f;
                g_individualTaskRun = true;
                g_runningTasks = true;
            } else if (g_cameraTasks[i].type == TaskType::SetZLength) {  // New Run handling for SetZLength
                if (g_cameraTasks[i].duration <= 0.0f)
                    g_cameraTasks[i].duration = 2.0f; // Ensure a valid duration

                g_individualStartWindowSize = g_windowSize;
                g_individualTargetWindowSize = g_cameraTasks[i].targetWindowSize;
                g_individualSegmentDuration = g_cameraTasks[i].duration;
                g_individualTaskType = TaskType::SetZLength;
                g_taskTimeAccumulator = 0.0f;
                g_individualTaskRun = true;
                g_runningTasks = true;
                g_currentTaskIndex = i;
            } else if (g_cameraTasks[i].type == TaskType::CameraRotate) { // ---- New: individual run handling
                g_individualTaskType = TaskType::CameraRotate;
                g_currentTaskIndex = i;
                g_taskTimeAccumulator = 0.0f;
                g_individualTaskRun = true;
                g_runningTasks = true; // Added: ensure the task is marked as active (green)
            } else if (g_cameraTasks[i].type == TaskType::SetSlomoFactor) {  // Updated branch.
                g_individualSegmentDuration = g_cameraTasks[i].duration;
                g_individualTaskType = TaskType::SetSlomoFactor;
                g_currentTaskIndex = i;
                g_taskTimeAccumulator = 0.0f;
                // Capture the start slomo factor immediately when scheduling the task.
                g_individualStartSlomo = g_slomoFactor;
                g_individualTaskRun = true;
                g_runningTasks = true; // This flag is needed for the task to show as running (green)
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Up") && i > 0) {
            std::swap(g_cameraTasks[i], g_cameraTasks[i - 1]);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Down") && i < g_cameraTasks.size() - 1) {
            std::swap(g_cameraTasks[i], g_cameraTasks[i + 1]);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Delete")) {
            g_cameraTasks.erase(g_cameraTasks.begin() + i);
            ImGui::PopID();
            continue;
        }
        
        // Show controls based on the task type.
        if (g_cameraTasks[i].type == TaskType::CameraMove) {
            ImGui::SliderFloat("Duration (s)", &g_cameraTasks[i].duration, 0.5f, 20.0f, "%.2f s");
        } else if (g_cameraTasks[i].type == TaskType::SetSimTime) {
            ImGui::SliderFloat("Target Sim Time (s)", &g_cameraTasks[i].simTime,                               g_startTimeMicroSeconds / 1e6f, g_endTimeMicroSeconds / 1e6f, "%.3f s");
        } else if (g_cameraTasks[i].type == TaskType::Wait) {
            ImGui::SliderFloat("Duration (s)", &g_cameraTasks[i].duration, 1.0f, 60.0f, "%.2f s");
        } else if (g_cameraTasks[i].type == TaskType::SetZLength) {
            // Add a slider to adjust the desired Z length (targetWindowSize) for the task.
            ImGui::SliderFloat("Target Z", &g_cameraTasks[i].targetWindowSize, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE, "%.2f");
            ImGui::SliderFloat("Duration (s)", &g_cameraTasks[i].duration, 0.5f, 20.0f, "%.2f s");
        } else if (g_cameraTasks[i].type == TaskType::CameraRotate) { // ---- New editing controls
            ImGui::SliderFloat("Rotation Angle (deg)", &g_cameraTasks[i].rotationAngle, -360.0f, 360.0f, "%.1f deg");
            ImGui::InputFloat3("Rotation Axis", &g_cameraTasks[i].rotationAxis[0]);
            ImGui::SliderFloat("Duration (s)", &g_cameraTasks[i].duration, 0.5f, 20.0f, "%.2f s");
        } else if (g_cameraTasks[i].type == TaskType::SetSlomoFactor) {  // ---- New editing controls
            ImGui::SliderFloat("Target Slomo Factor", &g_cameraTasks[i].targetSlomoFactor,
                               MIN_SLOMO_FACTOR, MAX_SLOMO_FACTOR, "%.2fx");
            ImGui::SliderFloat("Duration (s)", &g_cameraTasks[i].duration, 0.5f, 20.0f, "%.2f s");
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::EndChild();
        if (ImGui::Button("Start Render", ImVec2(0,0)) && g_cameraTasks.size() >= 1) {
        // Initiate render mode: reset buffers, update camera task state and begin recording.
        g_renderMode = true;
        g_videoProgress = 0.0;
        
        // Calculate total duration of all tasks
        g_totalTasksDuration = calculateTotalTasksDuration();
        g_elapsedTasksTime = 0.0f;
        
        // Swap width and height for vertical/portrait orientation based on user preference
        if (g_isVerticalLayout) {
            g_videoHeight = g_render4K ? 3840 : 1920;   // Height is now the larger dimension
            g_videoWidth = g_render4K ? 2160 : 1080;    // Width is now the smaller dimension
        } else {
            g_videoWidth = g_render4K ? 3840 : 1920;    // Width is now the larger dimension
            g_videoHeight = g_render4K ? 2160 : 1080;   // Height is now the smaller dimension
        }
        
        // Initialize ffmpeg pipe here
        initializeFFmpegPipe();
        
        g_isRecording = true;
        g_runningTasks = true;
        g_currentTaskIndex = 0;
        g_taskTimeAccumulator = 0.0f;
        g_startCamera = g_camera;
        g_individualTaskRun = false; // Ensure full-run mode.
    }
    ImGui::SameLine();
    if (ImGui::Button("Abort Render", ImVec2(0,0))) {
        // Cancel render mode: stop recording and close ffmpeg pipe
        g_renderMode = false;
        g_isRecording = false;
        closeFFmpegPipe();
        g_videoSaving = false;
        g_totalTasksDuration = 0.0f;
        g_elapsedTasksTime = 0.0f;
    }
    
    // Add 4K rendering option checkbox
    ImGui::Checkbox("Render in 4K", &g_render4K);

    if (g_renderMode) {
        if (g_videoSaving) {
            ImGui::ProgressBar(g_videoProgress, ImVec2(0.0f, 0.0f));
            ImGui::Text("Saving Video: %.0f%%", g_videoProgress * 100);
        } else {
            // Update elapsed time
            g_elapsedTasksTime = calculateElapsedTasksTime();
            
            // Display progress bar based on elapsed time vs total time
            float timeProgress = g_totalTasksDuration > 0.0f ? g_elapsedTasksTime / g_totalTasksDuration : 0.0f;
            ImGui::ProgressBar(timeProgress, ImVec2(0.0f, 0.0f));
            
            // Calculate and display estimated time to completion
            static auto renderStartTime = std::chrono::steady_clock::now();
            static bool firstRenderFrame = true;
            
            if (firstRenderFrame) {
                renderStartTime = std::chrono::steady_clock::now();
                firstRenderFrame = false;
            }
            
            if (timeProgress > 0.01f) {  // Only show estimate after we have some progress
                auto currentTime = std::chrono::steady_clock::now();
                float elapsedRealSeconds = std::chrono::duration<float>(currentTime - renderStartTime).count();
                
                // Calculate estimated total time based on current progress rate
                float estimatedTotalSeconds = elapsedRealSeconds / timeProgress;
                float remainingSeconds = estimatedTotalSeconds - elapsedRealSeconds;
                
                // Format time remaining in minutes and seconds
                int remainingMinutes = static_cast<int>(remainingSeconds) / 60;
                int remainingSecondsDisplay = static_cast<int>(remainingSeconds) % 60;
                
                ImGui::Text("Rendering: %.1f / %.1f seconds", g_elapsedTasksTime, g_totalTasksDuration);
                ImGui::Text("Estimated Completion: %dm %ds", remainingMinutes, remainingSecondsDisplay);
            } else {
                ImGui::Text("Rendering: %.1f / %.1f seconds", g_elapsedTasksTime, g_totalTasksDuration);
                ImGui::Text("Estimated Completion: Calculating...");
            }
        }
    } 

    // ----- New: Save and Load Task List via JSON -----
    ImGui::Separator();
    if (ImGui::Button("Save Tasks")) {
        saveTaskListToJson(std::string(g_taskListNameBuffer));
    }
    ImGui::Text("Available Task Lists");

    {
        namespace fs = std::filesystem;
        std::vector<std::string> availableTaskLists;
        fs::path taskFolder("task_lists");
        if (fs::exists(taskFolder) && fs::is_directory(taskFolder)) {
            for (const auto &entry : fs::directory_iterator(taskFolder)) {
                if (entry.is_regular_file()) {
                    std::string file = entry.path().filename().string();
                    // Only add files that end with ".json"
                    if (file.size() >= 6 && file.substr(file.size() - 5) == ".json") {
                        availableTaskLists.push_back(file);
                    }
                }
            }
        }
        // Optional: sort the list alphabetically.
        std::sort(availableTaskLists.begin(), availableTaskLists.end());

        // Static index to track the current selection.
        static int selectedTaskListIndex = -1;
        std::string currentComboDisplay = (selectedTaskListIndex >= 0 && selectedTaskListIndex < static_cast<int>(availableTaskLists.size()))
            ? availableTaskLists[selectedTaskListIndex]
            : "Select Task List";

        if (ImGui::BeginCombo("##TaskListDropdown", currentComboDisplay.c_str())) {
            for (int i = 0; i < static_cast<int>(availableTaskLists.size()); i++) {
                bool isSelected = (selectedTaskListIndex == i);
                if (ImGui::Selectable(availableTaskLists[i].c_str(), isSelected)) {
                    selectedTaskListIndex = i;
                    // DO NOT update g_taskListNameBuffer here.
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (ImGui::Button("Load Tasks")) {
            std::string taskListToLoad;
            if (selectedTaskListIndex >= 0 && selectedTaskListIndex < static_cast<int>(availableTaskLists.size()))
                taskListToLoad = availableTaskLists[selectedTaskListIndex];
            else
                taskListToLoad = std::string(g_taskListNameBuffer);

            // Remove the ".json" extension if present.
            if (taskListToLoad.size() >= 6 && taskListToLoad.substr(taskListToLoad.size() - 5) == ".json")
                taskListToLoad = taskListToLoad.substr(0, taskListToLoad.size() - 5);

            loadTaskListFromJson(taskListToLoad);
        }
    }
    // ---------------------------------------------------

    ImGui::Separator();
    ImGui::Text("Camera Pose:");
    ImGui::Text("Position: (%.1f x, %.1f y, %.1f z)", g_camera.center.x, g_camera.center.y, g_camera.center.z);
    ImGui::Text("Rotation: (%.1f° elev, %.1f° az, %.1f dist)", g_camera.elevation, g_camera.azimuth, g_camera.distance);
    if (ImGui::Button("Reset Camera"))
        resetCamera();

    // Button to toggle the visibility of the event volume box.
    if (ImGui::Button(g_showVolumeBox ? "Hide Volume Box" : "Show Volume Box"))
        g_showVolumeBox = !g_showVolumeBox;

    // New button to toggle the visibility of the coordinate axes.
    if (ImGui::Button(g_showCoordinateAxes ? "Hide Coordinate Axes" : "Show Coordinate Axes"))
        g_showCoordinateAxes = !g_showCoordinateAxes;

    ImGui::Separator();
    ImGui::Text("Color Settings:");
    if (ImGui::Button(g_randomizeColorsActive ? "Stop Randomize Colors" : "Start Randomize Colors")) {
        g_randomizeColorsActive = !g_randomizeColorsActive;
        if (g_randomizeColorsActive) {
            // On activation, initialize the transition using current colors as a base.
            g_positiveEventStartColor = g_positiveEventColor;
            g_negativeEventStartColor = g_negativeEventColor;
            
            // Generate a random hue between 0 and 360.
            float hue = static_cast<float>(rand()) / RAND_MAX * 360.0f;
            // Ensure full saturation and brightness (i.e. aggressive colors).
            glm::vec3 posColor = hsvToRgb(hue, 1.0f, 1.0f);
            // Use the complementary hue for maximum contrast.
            glm::vec3 negColor = hsvToRgb(fmod(hue + 180.0f, 360.0f), 1.0f, 1.0f);
            
            g_positiveEventTargetColor = glm::vec4(posColor, 1.0f);
            g_negativeEventTargetColor = glm::vec4(negColor, 1.0f);
            g_colorTransitionTimer = 0.0f;
        }
    }
    // Expose a slider to adjust the color transition duration (in seconds)
    ImGui::SliderFloat("Color Transition Duration (s)", &g_colorTransitionDuration, 0.1f, 10.0f, "%.1f s");
    
    float bgColor[4] = { g_backgroundColor.r, g_backgroundColor.g, g_backgroundColor.b, g_backgroundColor.a };
    if (ImGui::ColorEdit3("Background", bgColor))
        g_backgroundColor = glm::vec4(bgColor[0], bgColor[1], bgColor[2], 1.0f);
    
    float posColor[4] = { g_positiveEventColor.r, g_positiveEventColor.g, g_positiveEventColor.b, g_positiveEventColor.a };
    if (ImGui::ColorEdit3("Positive Events", posColor))
        g_positiveEventColor = glm::vec4(posColor[0], posColor[1], posColor[2], 1.0f);
    
    float negColor[4] = { g_negativeEventColor.r, g_negativeEventColor.g, g_negativeEventColor.b, g_negativeEventColor.a };
    if (ImGui::ColorEdit3("Negative Events", negColor))
        g_negativeEventColor = glm::vec4(negColor[0], negColor[1], negColor[2], 1.0f);

    ImGui::Separator();
    ImGui::Text("Metrics:");
    ImGui::Text("FPS: %.2f", avgFps);
    {
        std::lock_guard<std::mutex> lock(g_statsMutex);
        ImGui::Text("CPU Usage: %s", g_systemStats.cpuUsage.c_str());
        ImGui::Text("RAM Usage: %s", g_systemStats.ramUsage.c_str());
        ImGui::Text("GPU Usage: %s", g_systemStats.gpuUsage.c_str());
    }

    // -------------------------------------------------------
    // (Below: Existing code for event counts plotting)
    ImGui::Separator();
    ImGui::Text("Event Counts");
    
    size_t bufferedEvents;
    {
        std::lock_guard<std::mutex> lock(g_eventsMutex);
        bufferedEvents = g_prefetchedEvents.size();
    }

    static float values[2][90] = {0}; // Circular buffer for last 90 frames
    static int values_offset = 0;
    
    values[0][values_offset] = static_cast<float>(g_numPoints);
    values[1][values_offset] = static_cast<float>(bufferedEvents);
    values_offset = (values_offset + 1) % 90;
    
    float maxBuffered = 0;
    float maxVisualized = 0;
    for (int i = 0; i < 90; i++) {
        maxBuffered = std::max(maxBuffered, values[1][i]);
        maxVisualized = std::max(maxVisualized, values[0][i]); 
    }
    
    ImVec2 plotSize(0, 80);
    
    char bufferedLabel[32];
    char visualizedLabel[32];
    snprintf(bufferedLabel, sizeof(bufferedLabel), "Buffered (%.1fK)", bufferedEvents / 1e3);
    snprintf(visualizedLabel, sizeof(visualizedLabel), "Visualized (%.1fK)", g_numPoints / 1e3);
    
    ImGui::PlotLines("##Buffered", values[1], 90, values_offset, bufferedLabel, 0.0f, maxBuffered, plotSize);
    ImGui::PlotLines("##Events", values[0], 90, values_offset, visualizedLabel, 0.0f, maxVisualized, plotSize);

    // -------------------------------------------------------
    // NEW: Debug plot for simulation time.
    static float simTimeHistory[90] = {0};  // Circular buffer for sim time in seconds.
    static int simTime_offset = 0;
    simTimeHistory[simTime_offset] = g_simTimeMicroSeconds / 1e6f;  // Convert microseconds to seconds.
    simTime_offset = (simTime_offset + 1) % 90;
    float simMin = simTimeHistory[0], simMax = simTimeHistory[0];
    for (int i = 0; i < 90; i++) {
        simMin = std::min(simMin, simTimeHistory[i]);
        simMax = std::max(simMax, simTimeHistory[i]);
    }
    ImGui::PlotLines("Sim Time", simTimeHistory, 90, simTime_offset, "Sim Time (s)", simMin, simMax, plotSize);
    // -------------------------------------------------------

    // In the renderImGuiInterface function, add this section:
    ImGui::Separator();
    ImGui::Text("Graph Visualization:");
    if (ImGui::Checkbox("Show Event Graph", &g_showEventGraph)) {
        // Force update when toggling graph visibility
        if (g_showEventGraph) {
            updateEventBuffer(g_eventVBO, g_eventCBO);
        }
    }

    if (g_showEventGraph) {
        ImGui::SliderFloat("Connection Distance", &g_graphMaxDistance, 10.0f, 300.0f);
        ImGui::SliderInt("Max Connections", &g_graphMaxConnections, 1, 20);
        ImGui::SliderFloat("Line Width", &g_graphLineWidth, 0.1f, 5.0f);
        ImGui::SliderFloat("Connection Probability", &g_graphConnectionProb, 0.01f, 1.0f);
        ImGui::ColorEdit4("Line Color", &g_graphLineColor[0]);
        ImGui::Checkbox("Connect Same Polarity Only", &g_connectSamePolarity);
        ImGui::Checkbox("Connect Positive Events Only", &g_connectPositiveOnly);
        
        ImGui::Text("Graph Stats: %zu lines", g_numGraphLines);
    }
    
    ImGui::End(); // End the main "Controls" window.
    
    // Handle popup outside the main window
    if (displayPopup) {
        ImGui::OpenPopup("Render Complete");
    }
    
    // Position the popup in the center of the screen
    ImGui::SetNextWindowPos(ImVec2(windowWidth * 0.5f, windowHeight * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    
    if (ImGui::BeginPopupModal("Render Complete", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Video rendering completed successfully!");
        ImGui::Text("File saved to:");
        
        // Convert to absolute path and display it
        std::string absolutePath = std::filesystem::absolute(g_outputVideoPath).string();
        ImGui::TextWrapped("%s", absolutePath.c_str());
        
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            displayPopup = false;
        }
        ImGui::EndPopup();
    }
    
    // Imgui rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// Calculate the Model-View-Projection matrix.
glm::mat4 calculateMVP(int width, int height) {
    float aspect = width / static_cast<float>(height);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, MAX_WINDOW_SIZE);
    
    glm::mat4 view = glm::mat4(1.0f);
    view = glm::translate(view, glm::vec3(0, 0, -g_camera.distance));
    view = glm::rotate(view, glm::radians(g_camera.elevation), glm::vec3(1, 0, 0));
    view = glm::rotate(view, glm::radians(g_camera.azimuth), glm::vec3(0, 1, 0));
    view = glm::translate(view, -g_camera.center);
    
    // Apply volume rotation around the z-axis.
    // We use the rotation index to choose between 0°, 90°, 180° and 270°.
    float rotationAngle = 0.0f;
    switch (g_volumeRotationIndex) {
        case 1: rotationAngle = 90.0f; break;
        case 2: rotationAngle = 180.0f; break;
        case 3: rotationAngle = 270.0f; break;
        default: rotationAngle = 0.0f; break;
    }
    glm::vec3 volumeCenter = glm::vec3(EVENT_VOLUME_WIDTH / 2.0f, EVENT_VOLUME_HEIGHT / 2.0f, 0.0f);
    glm::mat4 model = glm::translate(glm::mat4(1.0f), volumeCenter)
                    * glm::rotate(glm::mat4(1.0f), glm::radians(rotationAngle), glm::vec3(0, 0, 1))
                    * glm::translate(glm::mat4(1.0f), -volumeCenter);
                    
    return projection * view * model;
}

// Setup buffers for the event volume box.
void setupEventVolumeBox() {
    glGenVertexArrays(1, &g_boxVAO);
    glBindVertexArray(g_boxVAO);

    glGenBuffers(1, &g_boxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, g_boxVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    std::vector<glm::vec4> boxColors(24, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    glGenBuffers(1, &g_boxColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, g_boxColorVBO);
    glBufferData(GL_ARRAY_BUFFER, boxColors.size() * sizeof(glm::vec4), 
                 boxColors.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// Draw the event volume box.
void drawEventVolumeBox(GLuint shaderProgram, const glm::mat4& MVP) {
    float halfZ = g_windowSize / 2.0f;  // Define half the z extent.
    std::vector<glm::vec3> boxVertices = {
        // Front face (z = +halfZ)
        glm::vec3(0.0f, 0.0f, halfZ),              glm::vec3(EVENT_VOLUME_WIDTH, 0.0f, halfZ),
        glm::vec3(EVENT_VOLUME_WIDTH, 0.0f, halfZ),  glm::vec3(EVENT_VOLUME_WIDTH, EVENT_VOLUME_HEIGHT, halfZ),
        glm::vec3(EVENT_VOLUME_WIDTH, EVENT_VOLUME_HEIGHT, halfZ), glm::vec3(0.0f, EVENT_VOLUME_HEIGHT, halfZ),
        glm::vec3(0.0f, EVENT_VOLUME_HEIGHT, halfZ), glm::vec3(0.0f, 0.0f, halfZ),
        
        // Back face (z = -halfZ)
        glm::vec3(0.0f, 0.0f, -halfZ),             glm::vec3(EVENT_VOLUME_WIDTH, 0.0f, -halfZ),
        glm::vec3(EVENT_VOLUME_WIDTH, 0.0f, -halfZ), glm::vec3(EVENT_VOLUME_WIDTH, EVENT_VOLUME_HEIGHT, -halfZ),
        glm::vec3(EVENT_VOLUME_WIDTH, EVENT_VOLUME_HEIGHT, -halfZ), glm::vec3(0.0f, EVENT_VOLUME_HEIGHT, -halfZ),
        glm::vec3(0.0f, EVENT_VOLUME_HEIGHT, -halfZ), glm::vec3(0.0f, 0.0f, -halfZ),
        
        // Connecting edges
        glm::vec3(0.0f, 0.0f, halfZ),  glm::vec3(0.0f, 0.0f, -halfZ),
        glm::vec3(EVENT_VOLUME_WIDTH, 0.0f, halfZ),  glm::vec3(EVENT_VOLUME_WIDTH, 0.0f, -halfZ),
        glm::vec3(EVENT_VOLUME_WIDTH, EVENT_VOLUME_HEIGHT, halfZ),  glm::vec3(EVENT_VOLUME_WIDTH, EVENT_VOLUME_HEIGHT, -halfZ),
        glm::vec3(0.0f, EVENT_VOLUME_HEIGHT, halfZ),  glm::vec3(0.0f, EVENT_VOLUME_HEIGHT, -halfZ)
    };

    glBindBuffer(GL_ARRAY_BUFFER, g_boxVBO);
    glBufferData(GL_ARRAY_BUFFER, boxVertices.size() * sizeof(glm::vec3), 
                 boxVertices.data(), GL_DYNAMIC_DRAW);

    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    
    glBindVertexArray(g_boxVAO);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, 24);
    glBindVertexArray(0);
}

// Render the complete 3D scene.
void renderScene(GLuint shaderProgram, GLuint vao, GLuint vbo, GLuint cbo, const glm::mat4& MVP) {
    glUseProgram(shaderProgram);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    
    // Draw the coordinate axes only if the flag is enabled.
    if (g_showCoordinateAxes) {
        drawCoordinateAxes(shaderProgram, MVP);
    }
    // Draw the event volume box only if the flag is enabled.
    if (g_showVolumeBox) {
        drawEventVolumeBox(shaderProgram, MVP);
    }
    
    // Draw the event graph before the points
    if (g_showEventGraph) {
        drawEventGraph(shaderProgram, MVP);
    }
    
    glUniform1f(glGetUniformLocation(shaderProgram, "pointSize"), g_pointSize);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glDrawArrays(GL_POINTS, 0, g_numPoints);
    glBindVertexArray(0);
}

// Function: updateCameraTasks
// UPDATED: Now the full-run mode for CameraMove tasks uses the current camera pose (stored in g_startCamera)
// as the start for the first CameraMove, and then for subsequent CameraMoves uses the previous camera's target.
void updateCameraTasks(float deltaTimeSeconds) {
    if (!g_runningTasks)
        return;
    
    // Individual mode:
    if (g_individualTaskRun) {
        if (g_individualTaskType == TaskType::CameraMove) {
            g_taskTimeAccumulator += deltaTimeSeconds;
            float t = g_taskTimeAccumulator / g_individualSegmentDuration;
            if (t > 1.0f)
                t = 1.0f;
            float smoothT = smootherstep(t);
            if (smoothT >= 1.0f) {
                g_camera = g_individualTrajectory.back();
                g_individualTaskRun = false;
                g_runningTasks = false;
                g_taskTimeAccumulator = 0.0f;
            } else {
                const Camera &startPose = g_individualTrajectory[0];
                const Camera &endPose = g_individualTrajectory[1];
                g_camera.center    = startPose.center    * (1.0f - smoothT) + endPose.center    * smoothT;
                g_camera.azimuth   = startPose.azimuth   * (1.0f - smoothT) + endPose.azimuth   * smoothT;
                g_camera.elevation = startPose.elevation * (1.0f - smoothT) + endPose.elevation * smoothT;
                g_camera.distance  = startPose.distance  * (1.0f - smoothT) + endPose.distance  * smoothT;
            }
        } else if (g_individualTaskType == TaskType::SetSimTime) {
            g_simTimeMicroSeconds = g_targetSimTime;
            g_individualTaskRun = false;
            g_runningTasks = false;
            g_taskTimeAccumulator = 0.0f;
        } else if (g_individualTaskType == TaskType::Wait) {
            g_taskTimeAccumulator += deltaTimeSeconds;
            if (g_taskTimeAccumulator >= g_individualSegmentDuration) {
                g_individualTaskRun = false;
                g_runningTasks = false;
                g_taskTimeAccumulator = 0.0f;
            }
        } else if (g_individualTaskType == TaskType::SetZLength) { 
            // On the first frame, capture the current z value as the start.
            if (g_taskTimeAccumulator == 0.0f)
                g_individualStartWindowSize = g_windowSize;
            
            g_taskTimeAccumulator += deltaTimeSeconds;
            float t = g_taskTimeAccumulator / g_individualSegmentDuration;
            if (t > 1.0f)
                t = 1.0f;
            float smoothT = smootherstep(t);
            g_windowSize = g_individualStartWindowSize * (1.0f - smoothT) + g_individualTargetWindowSize * smoothT;
            if (smoothT >= 1.0f) {
                g_windowSize = g_individualTargetWindowSize;
                g_individualTaskRun = false;
                g_runningTasks = false;
                g_taskTimeAccumulator = 0.0f;
            }
            return;
        } else if (g_individualTaskType == TaskType::CameraRotate) {
            CameraTask &task = g_cameraTasks[g_currentTaskIndex];
            g_taskTimeAccumulator += deltaTimeSeconds;
            float t = g_taskTimeAccumulator / task.duration;
            if (t > 1.0f)
                t = 1.0f;
            float smoothT = smootherstep(t);
            
            // Use event volume midpoint as the pivot.
            glm::vec3 pivot = glm::vec3(EVENT_VOLUME_WIDTH / 2.0f,
                                        EVENT_VOLUME_HEIGHT / 2.0f,
                                        0.0f);
            static glm::vec3 startEye;
            static bool startEyeInitialized = false;
            if (!startEyeInitialized) {
                float distance = g_camera.distance;
                float azimuth = glm::radians(g_camera.azimuth);
                float elevation = glm::radians(g_camera.elevation);
                glm::vec3 offset(distance * cos(elevation) * sin(azimuth),
                                 distance * sin(elevation),
                                 distance * cos(elevation) * cos(azimuth));
                // Compute the startEye relative to the fixed pivot.
                startEye = pivot - offset;
                startEyeInitialized = true;
            }
            
            glm::mat4 rotMatrix = glm::rotate(glm::mat4(1.0f),
                                              glm::radians(task.rotationAngle * smoothT),
                                              glm::normalize(task.rotationAxis));
            glm::vec3 newEye = pivot + glm::vec3(rotMatrix * glm::vec4(startEye - pivot, 1.0f));
            float newDistance = glm::length(newEye - pivot);
            glm::vec3 newOffset = pivot - newEye;
            float newElevation = glm::degrees(asin(newOffset.y / newDistance));
            float newAzimuth = glm::degrees(atan2(newOffset.x, newOffset.z));
            
            // Update only the rotation parameters.
            g_camera.distance = newDistance;
            g_camera.elevation = newElevation;
            g_camera.azimuth = newAzimuth;
            
            if (smoothT >= 1.0f) {
                g_currentTaskIndex++;
                g_taskTimeAccumulator = 0.0f;
                startEyeInitialized = false;
                g_individualTaskRun = false;
                g_runningTasks = false;
            }
            return;
        } else if (g_individualTaskType == TaskType::SetSlomoFactor) {
            // Updated individual-run branch for SetSlomoFactor.
            g_taskTimeAccumulator += deltaTimeSeconds;
            float t = (g_individualSegmentDuration > 0) ? (g_taskTimeAccumulator / g_individualSegmentDuration) : 1.0f;
            if (t > 1.0f)
                t = 1.0f;
            float smoothT = smootherstep(t);
            g_slomoFactor = g_individualStartSlomo * (1.0f - smoothT)
                           + g_cameraTasks[g_currentTaskIndex].targetSlomoFactor * smoothT;
            if (t >= 1.0f) {
                g_slomoFactor = g_cameraTasks[g_currentTaskIndex].targetSlomoFactor;
                g_individualTaskRun = false;
                g_runningTasks = false;
                g_taskTimeAccumulator = 0.0f;
            }
            return;
        }
        return;
    }
    
    // Full-run mode:
    if (g_currentTaskIndex >= g_cameraTasks.size()) {
        g_runningTasks = false;
        return;
    }
    
    CameraTask &curTask = g_cameraTasks[g_currentTaskIndex];
    
    if (curTask.type == TaskType::SetSimTime) {
        g_simTimeMicroSeconds = curTask.simTime * 1e6f;
        g_currentTaskIndex++;
        g_taskTimeAccumulator = 0.0f;
        return;
    } else if (curTask.type == TaskType::Wait) {
        g_taskTimeAccumulator += deltaTimeSeconds;
        if (g_taskTimeAccumulator >= curTask.duration) {
            g_taskTimeAccumulator = 0.0f;
            g_currentTaskIndex++;
        }
        return;
    } else if (curTask.type == TaskType::CameraMove) {
        // Only initialize the start position once for this CameraMove task.
        static int lastCameraMoveIndex = -1;
        static Camera fullRunStartCamera;
        if (g_currentTaskIndex != static_cast<size_t>(lastCameraMoveIndex)) {
            fullRunStartCamera = g_camera;
            lastCameraMoveIndex = g_currentTaskIndex;
        }

        g_taskTimeAccumulator += deltaTimeSeconds;
        float t = g_taskTimeAccumulator / curTask.duration;
        if (t > 1.0f)
            t = 1.0f;
        float smoothT = smootherstep(t);

        const Camera &endPose = curTask.camera;
        g_camera.center    = fullRunStartCamera.center    * (1.0f - smoothT) + endPose.center    * smoothT;
        g_camera.azimuth   = fullRunStartCamera.azimuth   * (1.0f - smoothT) + endPose.azimuth   * smoothT;
        g_camera.elevation = fullRunStartCamera.elevation * (1.0f - smoothT) + endPose.elevation * smoothT;
        g_camera.distance  = fullRunStartCamera.distance  * (1.0f - smoothT) + endPose.distance  * smoothT;

        if (smoothT >= 1.0f) {
            g_camera = endPose;
            g_currentTaskIndex++;
            g_taskTimeAccumulator = 0.0f;
            // Reset for next CameraMove task.
            lastCameraMoveIndex = -1;
        }
    } else if (curTask.type == TaskType::SetZLength) {
        static int lastTaskIndex = -1;
        static float fullRunStartZ = 0.0f;
        if (g_currentTaskIndex != lastTaskIndex) {
            fullRunStartZ = g_windowSize;
            lastTaskIndex = g_currentTaskIndex;
        }
        
        g_taskTimeAccumulator += deltaTimeSeconds;
        float t = g_taskTimeAccumulator / curTask.duration;
        if (t > 1.0f)
            t = 1.0f;
        float smoothT = smootherstep(t);
        g_windowSize = fullRunStartZ * (1.0f - smoothT) + curTask.targetWindowSize * smoothT;
        if (smoothT >= 1.0f) {
            g_windowSize = curTask.targetWindowSize;
            g_currentTaskIndex++;
            g_taskTimeAccumulator = 0.0f;
            lastTaskIndex = -1;
        }
        return;
    } else if (curTask.type == TaskType::CameraRotate) {
        static glm::vec3 fullRunStartEye;
        static int lastTaskIndex = -1;
        glm::vec3 pivot = glm::vec3(EVENT_VOLUME_WIDTH / 2.0f,
                                    EVENT_VOLUME_HEIGHT / 2.0f,
                                    0.0f);
        if (g_currentTaskIndex != static_cast<size_t>(lastTaskIndex)) {
            float distance = g_camera.distance;
            float azimuth = glm::radians(g_camera.azimuth);
            float elevation = glm::radians(g_camera.elevation);
            glm::vec3 offset(distance * cos(elevation) * sin(azimuth),
                             distance * sin(elevation),
                             distance * cos(elevation) * cos(azimuth));
            fullRunStartEye = pivot - offset;
            lastTaskIndex = g_currentTaskIndex;
        }
        
        g_taskTimeAccumulator += deltaTimeSeconds;
        float t = g_taskTimeAccumulator / curTask.duration;
        if (t > 1.0f)
            t = 1.0f;
        float smoothT = smootherstep(t);
        
        glm::mat4 rotMatrix = glm::rotate(glm::mat4(1.0f),
                                          glm::radians(curTask.rotationAngle * smoothT),
                                          glm::normalize(curTask.rotationAxis));
        glm::vec3 newEye = pivot + glm::vec3(rotMatrix * glm::vec4(fullRunStartEye - pivot, 1.0f));
        float newDistance = glm::length(newEye - pivot);
        glm::vec3 newOffset = pivot - newEye;
        float newElevation = glm::degrees(asin(newOffset.y / newDistance));
        float newAzimuth = glm::degrees(atan2(newOffset.x, newOffset.z));
        
        g_camera.distance = newDistance;
        g_camera.elevation = newElevation;
        g_camera.azimuth = newAzimuth;
        
        if (smoothT >= 1.0f) {
            g_currentTaskIndex++;
            g_taskTimeAccumulator = 0.0f;
            lastTaskIndex = -1;
        }
        return;
    } else if (curTask.type == TaskType::SetSlomoFactor) {
        static int lastTaskIndex = -1;
        static float fullRunStartSlomo = g_slomoFactor;
        if (g_currentTaskIndex != static_cast<size_t>(lastTaskIndex)) {
            fullRunStartSlomo = g_slomoFactor;
            lastTaskIndex = g_currentTaskIndex;
        }
        g_taskTimeAccumulator += deltaTimeSeconds;
        float t = g_taskTimeAccumulator / curTask.duration;
        if (t > 1.0f)
            t = 1.0f;
        float smoothT = smootherstep(t);
        g_slomoFactor = fullRunStartSlomo * (1.0f - smoothT)
                        + curTask.targetSlomoFactor * smoothT;
        if (smoothT >= 1.0f) {
            g_slomoFactor = curTask.targetSlomoFactor;
            g_currentTaskIndex++;
            g_taskTimeAccumulator = 0.0f;
            lastTaskIndex = -1;
        }
        return;
    }
}




void updateColorRandomization(float deltaTimeSeconds) {
    if (!g_randomizeColorsActive)
        return;
        
    g_colorTransitionTimer += deltaTimeSeconds;
    float t = g_colorTransitionTimer / g_colorTransitionDuration;  // Use the modifiable duration
    t = (t > 1.0f) ? 1.0f : t;
    // Smootherstep interpolation: t^3( t*(6*t-15)+10 )
    float smoothT = t * t * t * (t * (6.0f * t - 15.0f) + 10.0f);
    
    // Interpolate between the start and target colors.
    g_positiveEventColor = glm::mix(g_positiveEventStartColor, g_positiveEventTargetColor, smoothT);
    g_negativeEventColor = glm::mix(g_negativeEventStartColor, g_negativeEventTargetColor, smoothT);
    
    // Once the transition is complete, choose new random target colors.
    if (g_colorTransitionTimer >= g_colorTransitionDuration) {
        g_positiveEventStartColor = g_positiveEventTargetColor;
        g_negativeEventStartColor = g_negativeEventTargetColor;
        // Generate a random hue between 0 and 360.
        float hue = static_cast<float>(rand()) / RAND_MAX * 360.0f;
        // Ensure full saturation and brightness (i.e. aggressive colors).
        glm::vec3 posColor = hsvToRgb(hue, 1.0f, 1.0f);
        // Use the complementary hue for maximum contrast.
        glm::vec3 negColor = hsvToRgb(fmod(hue + 180.0f, 360.0f), 1.0f, 1.0f);
        
        g_positiveEventTargetColor = glm::vec4(posColor, 1.0f);
        g_negativeEventTargetColor = glm::vec4(negColor, 1.0f);
        g_colorTransitionTimer = 0.0f;
    }
}

int main(int argc, char *argv[]) {
    // Add this near the start of main() function, before the window creation
    std::atexit([]() {
        g_stopLoading = true;
        if (g_loaderThread.joinable()) {
            g_loaderThread.join();
        }
    });

    // Remove this early update of HDF5 files
    // updateHDF5FileList();
    
    // Initialize with empty paths - will be set properly later
    g_selectedFileIndex = -1;
    g_eventFilePath = "";
    
    // Set default simulation time values
    g_simTimeMicroSeconds = 0.0f;
    g_simTimeMicroSecondsTask = 0.0f;
    
    resetCamera();
    
    // We'll start the loader thread after setting up the paths properly
    // No need for this check here
    /* 
    if (!g_eventFilePath.empty()) {
        loaderThread = std::thread(continuousLoadEventsFromHDF5, g_eventFilePath);
    }
    */
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, 
        "Realtime 3D Event Visualization (C++)", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    
    GLuint shaderProgram = createShaderProgram();
    
    GLuint vao;
    GLuint vbo, cbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);
    
    glGenBuffers(1, &cbo);
    glBindBuffer(GL_ARRAY_BUFFER, cbo);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    
    g_lastTimeSeconds = glfwGetTime();
    double accumulator = 0.0;
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    setupCoordinateAxes();
    setupEventVolumeBox();
    
    std::thread monitorThread(systemMonitoringThread);
    std::cout << "System monitoring thread started." << std::endl;

    // Get the executable path and set app_dir to two levels up
    std::filesystem::path executable_path = std::filesystem::absolute(argv[0]);
    std::filesystem::path app_dir = executable_path.parent_path().parent_path();
    
    // Define data, task_lists, and render directories
    std::filesystem::path data_dir = app_dir / "data";
    std::filesystem::path task_lists_dir = app_dir / "task_lists";
    std::filesystem::path render_dir = app_dir / "render";
    
    // Create directories if they don't exist
    if (!std::filesystem::exists(data_dir)) {
        std::filesystem::create_directory(data_dir);
    }
    
    if (!std::filesystem::exists(task_lists_dir)) {
        std::filesystem::create_directory(task_lists_dir);
    }
    
    if (!std::filesystem::exists(render_dir)) {
        std::filesystem::create_directory(render_dir);
    }
    
    std::cout << "Using application directory: " << app_dir.string() << std::endl;
    std::cout << "Data directory: " << data_dir.string() << std::endl;
    std::cout << "Task lists directory: " << task_lists_dir.string() << std::endl;
    std::cout << "Render directory: " << render_dir.string() << std::endl;
    
    // Update the global data directory variable
    g_dataDir = data_dir.string();
    
    // NOW scan for HDF5 files in the auto-determined data directory
    updateHDF5FileList();
    
    // Set default event file path if HDF5 files are found
    if (!g_hdf5Files.empty()) {
        g_selectedFileIndex = 0;
        g_eventFilePath = (std::filesystem::path(g_dataDir) / g_hdf5Files[0]).string();
        std::cout << "Using event file: " << g_eventFilePath << std::endl;
        
        // Now start the loader thread with the correct file path
        g_loaderThread = std::thread(continuousLoadEventsFromHDF5, g_eventFilePath);
        
        // Update min/max event times from the selected file
        setStartEndTime(g_eventFilePath);
    }
    
    while (!glfwWindowShouldClose(window)) {
        double currentTimeSeconds = glfwGetTime();
        double rawdeltaTimeSecondsSeconds = currentTimeSeconds - g_lastTimeSeconds;
        double deltaTimeSeconds;

        // Use a fixed timestep in render mode to ensure full simulation duration.
        if (g_renderMode) {
            deltaTimeSeconds = 1.0 / 60.0;
        } else {
            deltaTimeSeconds = std::min(rawdeltaTimeSecondsSeconds, 0.033);
        }
        g_lastTimeSeconds = currentTimeSeconds;

        accumulator += deltaTimeSeconds;
        double avgFps;
        updateFPS(deltaTimeSeconds, avgFps);
        glfwPollEvents();
        
        // clearing buffers in the main loop avoids segfaults when running resetApplication
        if (g_shouldClearBuffers) {
            clearEventBuffers(g_eventVBO, g_eventCBO);
            g_shouldClearBuffers = false;
        }

        updateSimulation(deltaTimeSeconds, vbo, cbo);
        updateCameraTasks(deltaTimeSeconds);
        updateColorRandomization(deltaTimeSeconds);
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        int windowWidth, windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        
        // Query the window content scale:
        float xscale = 1.0f, yscale = 1.0f;
        glfwGetWindowContentScale(window, &xscale, &yscale);
        int guiWidthPhysical = static_cast<int>(g_GUI_FIXED_WIDTH * xscale);

        int viewportWidth = width - guiWidthPhysical; 
        if (viewportWidth < 0)
            viewportWidth = width;
        
        glClearColor(g_backgroundColor.r, g_backgroundColor.g, g_backgroundColor.b, g_backgroundColor.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glViewport(0, 0, viewportWidth, height);
        glm::mat4 MVP = calculateMVP(viewportWidth, height);
        renderScene(shaderProgram, vao, vbo, cbo, MVP);

        // If we are in render mode, capture the rendered frame.
        if (g_renderMode && g_isRecording) {
            // Store original viewport dimensions
            int originalWidth, originalHeight;
            glfwGetFramebufferSize(window, &originalWidth, &originalHeight);
            
            // Calculate the viewport width (excluding ImGui area)
            float xscale = 1.0f, yscale = 1.0f;
            glfwGetWindowContentScale(window, &xscale, &yscale);
            int guiWidthPhysical = static_cast<int>(g_GUI_FIXED_WIDTH * xscale);
            int viewportWidth = originalWidth - guiWidthPhysical;
            if (viewportWidth < 0)
                viewportWidth = originalWidth;
            
            int captureWidth, captureHeight;
            if (g_render4K) {
                // Set dimensions for true 4K
                if (g_isVerticalLayout) {
                    captureWidth = 2160;  // 9:16 aspect ratio for vertical
                    captureHeight = 3840;
                } else {
                    captureWidth = 3840;  // 16:9 aspect ratio for horizontal
                    captureHeight = 2160;
                }
            } else {
                // Use fixed HD dimensions for non-4K mode
                if (g_isVerticalLayout) {
                    captureWidth = 1080;  // 9:16 aspect ratio for vertical
                    captureHeight = 1920;
                } else {
                    captureWidth = 1920;  // 16:9 aspect ratio for horizontal
                    captureHeight = 1080;
                }
            }
            
            // Initialize video dimensions on first capture
            if (g_videoWidth == 0 || g_videoHeight == 0) {
                g_videoWidth = captureWidth;
                g_videoHeight = captureHeight;
            }
            
            // For 4K rendering or if we're only capturing the viewport
            GLuint fbo = 0, renderTexture = 0;
            
            // Create a framebuffer for correct resolution rendering
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            
            // Create a color attachment texture
            glGenTextures(1, &renderTexture);
            glBindTexture(GL_TEXTURE_2D, renderTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, captureWidth, captureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);
            
            // Create a renderbuffer for depth
            GLuint rbo;
            glGenRenderbuffers(1, &rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, captureWidth, captureHeight);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
            
            // Set the viewport to the capture dimensions
            glViewport(0, 0, captureWidth, captureHeight);
            
            // Clear and render the scene at the correct resolution
            glClearColor(g_backgroundColor.r, g_backgroundColor.g, g_backgroundColor.b, g_backgroundColor.a);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            // Calculate MVP with the correct aspect ratio
            glm::mat4 MVP = calculateMVP(captureWidth, captureHeight);
            
            // Render the scene
            renderScene(shaderProgram, vao, vbo, cbo, MVP);
            
            // Allocate buffer for frame data
            std::vector<unsigned char> frameBuffer(captureWidth * captureHeight * 3);
            
            // Read pixels from the framebuffer
            glReadPixels(0, 0, captureWidth, captureHeight, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer.data());
            
            // Either write directly to ffmpeg pipe or store in memory 
            if (g_renderMode && g_ffmpegPipe) {
                // Write directly to ffmpeg pipe
                writeFrameToFFmpeg(frameBuffer);
                
                // Update progress based on task index (frame count) for compatibility
                g_videoProgress = g_currentTaskIndex / static_cast<float>(g_cameraTasks.size());
                
                // Update elapsed time for the time-based progress
                g_elapsedTasksTime = calculateElapsedTasksTime();
            }
            
            // Clean up rendering resources
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteTextures(1, &renderTexture);
            glDeleteFramebuffers(1, &fbo);
            
            // Restore original viewport
            glViewport(0, 0, originalWidth, originalHeight);
        }
        // Once the task list execution is complete we stop recording and close the video
        if (g_renderMode && !g_runningTasks && g_isRecording) {
            g_isRecording = false;
            g_videoSaving = false;
            g_renderMode = false;
            // Close the ffmpeg pipe to finalize the video
            closeFFmpegPipe();
            // Reset time tracking variables
            g_totalTasksDuration = 0.0f;
            g_elapsedTasksTime = 0.0f;
            // Set flag to show render complete popup
            g_showRenderCompletePopup = true;
        }

        glViewport(0, 0, width, height);
        renderImGuiInterface(avgFps, windowWidth, windowHeight);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    g_stopLoading = true;
    if (g_loaderThread.joinable()) {
        g_stopLoading = true;
        g_loaderThread.join();
    }
    
    g_shouldStopMonitoring = true;
    if (monitorThread.joinable())
        monitorThread.join();

    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &cbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();

    glDeleteBuffers(1, &g_axisVBO);
    glDeleteBuffers(1, &g_axisColorVBO);
    glDeleteVertexArrays(1, &g_axisVAO);

    glDeleteBuffers(1, &g_boxVBO);
    glDeleteBuffers(1, &g_boxColorVBO);
    glDeleteVertexArrays(1, &g_boxVAO);

    // Make sure to clean up the graph buffers in the cleanup section
    // Add this in your cleanup code
    if (g_graphVAO != 0) {
        glDeleteVertexArrays(1, &g_graphVAO);
        g_graphVAO = 0;
    }
    if (g_graphVBO != 0) {
        glDeleteBuffers(1, &g_graphVBO);
        g_graphVBO = 0;
    }
    if (g_graphColorVBO != 0) {
        glDeleteBuffers(1, &g_graphColorVBO);
        g_graphColorVBO = 0;
    }

    return 0;
}
