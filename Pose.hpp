
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>

#include "AppState.hpp"

void updatePose()
{
    AppState& s = AppState::get();
    assert(s.poses.has_value());

    cnpy::NpyArray& arr = s.poses.value();
    double* data = arr.data<double>();

    if (s.enable_trajectory) {
        double *p = data + static_cast<size_t>(s.pose_idx) * arr.shape[1];

        /*  NOTE: traj.py uses X forward, Y right, Z down for the camera.
            However, we are using -Z forward, X right, Y up for the camera.
            So, in order to get the same rendering effect as the ones
            produced by traj.py, we will negate the yaw values here.

            NOTE: determine if this affects data labeling / optimization?
            Initial assessment appears to suggest no, since we optimize on
            the relative rotation between frames (i.e. setting the rotation
            matrix of frame 1 to identity).

            We just have to set up the same camera model in optimization
            (i.e. -Z forward, X right, Y up)
        */
        float yaw = -1 * static_cast<float>(p[0]);  // pan
        float pitch = static_cast<float>(p[1]);     // tilt
        float roll = static_cast<float>(p[2]);
        float fov = static_cast<float>(p[3]);

        s.pan = yaw, s.tilt = pitch, s.roll = roll;
        s.fov = fov;

        glm::mat4 R {1.0f};
        /*  NOTE: instead of yawing around the global Y after pitching,
            we can simply yaw before pitching when the local/global Y are
            still coincident.
        */
        R = glm::rotate(R, glm::radians(yaw), {0, 1, 0});
        R = glm::rotate(R, glm::radians(pitch), {1, 0, 0});
        // R = glm::rotate(R, glm::radians(yaw), glm::vec3(glm::row(R, 1)));
        R = glm::rotate(R, glm::radians(roll), {0, 0, -1});
        s.M_rot = R;
        s.pose_idx = static_cast<int>(static_cast<size_t>(s.pose_idx + 1) % arr.shape[0]);
    }
}
