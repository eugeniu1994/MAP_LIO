 if camera == 'basler':
            # intrinsic matrix
            K = np.array([[1363.18778612,    0.,         978.05715426],
                          [ 0.,           1362.77700097, 607.65710195],
                          [ 0.,              0.,           1.        ]])
            # distortion
            D = np.array([-0.15425933, 0.13876932, -0.00066874, 0.00093961, -0.06687253])

            # extrinsic params
            rvec = np.array([0.,0.,0.])
            rot_mat = None
            tvec = np.array([0.,0.,0.])
        elif camera == 'thermal_center':
            # intrinsic matrix
            K = np.array([[1495.84601928,    0.,          355.3401074 ],
                          [   0.,         1495.58217734,  257.10045874],
                          [   0.,            0.,            1.        ]])
            # distortion
            D = np.array([-0.13991353, -1.62366771, 0.00110071, 0.00492267, 3.95121079])

            # extrinsic params
            rvec = None
            rot_mat = cv2.Rodrigues(np.array([-0.00363865, 0.00427818, 0.0104788]))[0]
            tvec = np.array([-0.01349176, 0.06057121, 0.05655506])
        elif camera == 'thermal_right':
            # intrinsic matrix
            K = np.array([[1499.29684013  ,  0.    ,      338.76978523],
                          [   0.     ,    1500.27024765,  239.59764372],
                          [   0.     ,       0.    ,        1.        ]])
            # distortion
            D = np.array([-0.13991353, -1.62366771, 0.00110071, 0.00492267, 3.95121079])

            # extrinsic params
            rvec = None
            rot_mat = cv2.Rodrigues(np.array([-0.003661 , 0.4046488 , 0.003575]))[0]
            tvec = np.array([0.04314681, 0.05490265, 0.02144468])
        elif camera == 'thermal_left':
            # intrinsic matrix
            K = np.array([[1496.0431082,    0.,         349.23737282],
                          [   0.,        1497.37541302, 238.32624878],
                          [   0.,           0.,           1.        ]])
            # distortion
            D = np.array([-0.13991353, -1.62366771, 0.00110071, 0.00492267, 3.95121079])

            # extrinsic params
            rvec = None
            rot_mat = cv2.Rodrigues(np.array([0.0006695, -0.41386521, 0.01294857]))[0]
            tvec = np.array([-0.0569287, 0.05887872, 0.03450623])



  lidar_cam_rot = np.array([[0.999883,    0.00902631,  0.0123187],
                                  [0.0131557 , -0.0994464 , -0.994956 ],
                                  [-0.00775573, 0.995002  , -0.0995536]])
        #lidar_cam_rot = np.matmul(lidar_cam_rot, eulerAnglesToRotationMatrix(np.array([-0.005,0.0,-0.0013]))) # NOTE: A fix by Jyri
        lidar_cam_t =  np.array([[0.00286508],
                                 [-0.0854374],
                                 [-0.427633]])