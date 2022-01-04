import cv2
import time
import mediapipe as mp
from numpy import dot
from numpy.linalg import norm

# Call Module for Media Pipe 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Score
avg_score = "None"

# Cosine Similarity
def findCosineSimilarity_1(A, B):
    return dot(A, B)/(norm(A)*norm(B))

# Evaluate User's Motion
def compare_positions(trainer_video, user_video, keyp_list, dim = (420,720)):
    global avg_score

    fps_time = 0 
    key_ = 0
    tot_score=[]
    len_tot=1

    cap = cv2.VideoCapture(trainer_video) # Dance Detect Cam
    cam = user_video # Live Cam

    cam.set(3, 646)
    cam.set(4, 364)

    # Dance Detection
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened() and cam.isOpened():
            ret_1, frame_1 = cam.read()
            ret_2, frame_2 = cap.read()
            
            # Preprocessing Video
            if frame_1 is not None and frame_2 is not None:
                # Recolor Feed
                image1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
                image2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)
                
                # Displaying the dancer feed
                image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
                cv2.imshow('Dancer Window', image2)
                
                # Recolor image back to BGR for rendering
                image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                image1 = cv2.flip(image1, 1)

                results1 = holistic.process(image1)
                height, width, _ = image1.shape

                # Calculate the Cosine Similarity
                if results1.pose_landmarks:
                    points = results1.pose_landmarks.landmark
                    features = []
                    for i in range(0, len(points)):
                        try:
                            features.append(int(points[i].x * width + 0.5))
                            features.append(int(points[i].y * width + 0.5))
                        except:
                            features.append(0)
                            features.append(0)

                    sim_score = findCosineSimilarity_1(keyp_list[key_ * 33],features)
                    key_ += 1

                    # Displaying the minimum cosine score
                    cv2.putText(image1, str(sim_score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # If the disctance is below the threshold
                    if 0.98 <= sim_score <= 1:
                        cv2.putText(image1, "SCORE : " + str(int(sum(tot_score)/len_tot*100)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        tot_score.append(1)
                    else:
                        cv2.putText(image1, "SCORE : " + str(int(sum(tot_score)/len_tot*100)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        tot_score.append(0)
                    cv2.putText(image1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    len_tot=len(tot_score)

                # Render detections
                mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color = (211, 203, 197), thickness = 2, circle_radius = 2), 
                                        mp_drawing.DrawingSpec(color = (159, 106, 141), thickness = 2, circle_radius = 2) 
                                         )      
        
                # Display the user feed
                cv2.imshow('User Window', image1)

                fps_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cam.release()
        cap.release()
        cv2.destroyAllWindows()
        avg_score = (sum(tot_score) / len_tot) * 100
        print(avg_score)