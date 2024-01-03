
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "IndexBuffer.hpp"
#include "Shader.hpp"
#include "Sphere.hpp"
#include "VertexBuffer.hpp"
#include "Window.hpp"
#include "cube.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "utils.hpp"

static void glErrorCallback_(GLenum source, GLenum type, GLuint id,
                             GLenum severity, GLsizei length, const GLchar* msg,
                             const void* userParam)
{
    LOG("OpenGL error:");
    LOG(msg);
}

void keyCallback_(GLFWwindow* window, int key, int scancode, int action,
                  int mods);
void processInput(GLFWwindow* window);

float fov = 75.0f;
float max_fov = 120.0f;
bool showUI = true;

// Initial camera orientation: upright, facing negative z axis
// clang-format off
float init_rot_M[] = {
    1,  0,  0,  0,
    0,  1,  0,  0,
    0,  0, -1,  0,
    0,  0,  0,  1,
};
// clang-format on
// Transpose, because GLM is column-major by default
glm::mat4 M_rot = glm::transpose(glm::make_mat4(init_rot_M));

// "front" is the intrinsic (local) -z axis.
glm::vec3 front = {0, 0, -1};
glm::vec3 up, right;


float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main(int argc, char** argv)
{
    const int w_w = 1280, w_h = 720;
    Window window(w_w, w_h, "OpenGL Test", argc <= 3);
    glfwSetKeyCallback(window.get(), keyCallback_);
    glfwSwapInterval(1);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window.get(), true);
    ImGui_ImplOpenGL3_Init("#version 150");

    if (glewInit() != GLEW_OK) throw std::runtime_error("GLEW init failed.");

    if (GLEW_KHR_debug)
        glDebugMessageCallback(glErrorCallback_, nullptr);
    else
        LOG("glDebugMessageCallback not available.");

    std::string filePath = argc < 2 ? "../images/p1.jpg" : argv[1];
    int img_w, img_h, img_channels;
    stbi_set_flip_vertically_on_load(true);
    uint8_t* img =
        stbi_load(filePath.c_str(), &img_w, &img_h, &img_channels, 0);

    if (!img) throw std::runtime_error("stbi_load() failed!");
    assert(img_channels == 3);

    uint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = {0.2f, 0.2f, 0.2f, 1.0f};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_w, img_h, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, img);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(img);

    // Rectangle (two triangles) covering the screen.
    constexpr int stride = 3;
    // clang-format off
    std::vector<float> vertices {
        -1, -1,  0,
         1,  1,  0,
        -1,  1,  0,
        -1, -1,  0,
         1, -1,  0,
         1,  1,  0,
    };
    // clang-format on

    VertexBuffer vb(vertices.data(),
                    static_cast<uint>(vertices.size() * sizeof(float)));

    Shader shader(utils::path("shaders/vertex.glsl"),
                  utils::path("shaders/frag.glsl"));
    shader.use();
    {
        // proportion of the missing sphere that's below the horizon
        double m = 1;
        if (argc >= 3) m = std::stod(argv[2]);
        double u = (double)img_w / 2 / (double)img_h;
        float v_n = (float)(u * M_1_PI);
        float v_b = (float)(u / 2 + (1 - u) * m);
        shader.setFloat("v_norm", v_n);
        shader.setFloat("v_bias", v_b);
    }

    uint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float),
                          (void*)0);

    vb.bind();
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

    int frame_cnt = 0;
    float fps_sum = 0;
    float fps = 1;

    while (glfwWindowShouldClose(window.get()) == 0) {
        // Compute fps
        frame_cnt++;
        fps_sum += io.Framerate;
        {
            int limit = (int)fps / 2 + 1;
            if (frame_cnt == limit) {
                fps = fps_sum / (float)limit;
                frame_cnt = 0;
                fps_sum = 0;
            }
        }

        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Process input
        glfwPollEvents();
        processInput(window.get());

        // Clear frame
        glClear(GL_COLOR_BUFFER_BIT);

        // Recalculate FoV, LoD, perspective, & view.
        float fov_thresh = 75.0f;
        if (fov <= fov_thresh)
            shader.setFloat("lod", 0.0f);
        else
            shader.setFloat("lod",
                            (fov - fov_thresh) / (max_fov - fov_thresh) + 1.0f);

        glm::mat4 M_proj = glm::perspective(
            glm::radians(fov), (float)w_w / (float)w_h, 0.1f, 2.0f);
        shader.setMat4("proj", M_proj);

        glm::mat4 M_view =
            glm::lookAt(glm::vec3(0), glm::vec3(glm::column(M_rot, 2)),
                        glm::vec3(glm::column(M_rot, 1)));

        shader.setMat4("view", M_view);

        // Draw projected panorama
        glBindTexture(GL_TEXTURE_2D, tex);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0,
                     static_cast<int>(vertices.size() / stride));

        // Declare UI
        if (showUI) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            {
                // ImGui::SetNextWindowSize(ImVec2(270, 80));
                ImGui::Begin("Debug Info (Press H to hide/show)", nullptr,
                             ImGuiWindowFlags_NoFocusOnAppearing |
                                 ImGuiWindowFlags_NoScrollbar);
                ImGui::Text("Pitch: %.1f°, Yaw: %.1f°, FoV: %.0f°", 0.0f, 0.0f,
                            fov);
                ImGui::Text("Average %.2f ms/frame (%.1f FPS)", 1000.0f / fps,
                            fps);
                // ImGui::Text("Window focused: %s", ImGui::IsWindowFocused());
                ImGui::Text("M_rot: ");
                ImGui::SameLine();
                ImGui::Text("%s",
                            utils::pretty_matrix(glm::value_ptr(M_rot), 4, 4, 2)
                                .c_str());
                ImGui::Text("|right|: %f", glm::length(right));
                ImGui::End();
            }

            // Render UI
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glfwSwapBuffers(window.get());

        if (!window.visible()) {
            /*
            Swap again so both the front and back buffer is guaranteed to
            contain the latest frame, which is then read by glReadPixels
            below. If we omit this, glReadPixels may return an empty frame on
            some platforms (because it read the other buffer which was not
            rendered to yet).
            */
            glfwSwapBuffers(window.get());
            break;
        }
    }

    int w, h;
    glfwGetFramebufferSize(window.get(), &w, &h);
    LOG("WH: ", w, " ", h);

    auto imgOut = std::make_unique<uint8_t[]>(static_cast<size_t>(w * h * 3));
    assert(imgOut.get());

    // glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, imgOut.get());
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg("out.jpg", w, h, 3, imgOut.get(), 90);

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    return 0;
}

void keyCallback_(GLFWwindow* window, int key, int scancode, int action,
                  int mods)
{
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_H) showUI = !showUI;
    }
}

void processInput(GLFWwindow* window)
{
    // "up" is the extrinsic (global) y axis.
    // row of M_rot is equiv. to col of inverse(M_rot).
    up = glm::row(M_rot, 1);

    // intrinsic (local) x axis.
    right = {1, 0, 0};

    // normalized projection of the intrinsic x axis
    // onto the extrinsic xz plane.
    // This has a problem: when the camera looks straight up or down,
    // up and front will be parallel, resulting in a cross product with
    // zero magnitude, getting the camera stuck.
    //right = glm::cross(front, up);

    glm::vec3 right_ = glm::normalize(right);

    float fov_a = 40 * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) fov -= fov_a;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) fov += fov_a;

    fov = std::min(max_fov, fov);
    fov = std::max(10.0f, fov);

    float rot_a = 1.2f - (max_fov - fov) / max_fov;
    rot_a *= deltaTime * 40;

    // Passive rotation, so the rotation amount is the
    // negative of the desired rotation direction.
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        M_rot = glm::rotate(M_rot, glm::radians(-rot_a), front);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        M_rot = glm::rotate(M_rot, glm::radians(rot_a), front);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        M_rot = glm::rotate(M_rot, glm::radians(-rot_a), right_);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        M_rot = glm::rotate(M_rot, glm::radians(rot_a), right_);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        M_rot = glm::rotate(M_rot, glm::radians(-rot_a), up);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        M_rot = glm::rotate(M_rot, glm::radians(rot_a), up);
}
