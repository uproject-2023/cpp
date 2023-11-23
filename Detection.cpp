#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <map>
#include <iostream>
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <mach-o/dyld.h>
#include <climits>
#else
#include <unistd.h>
#endif
#include <experimental/filesystem>


const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.6;
const float NMS_THRESHOLD = 0.55;
const float CONFIDENCE_THRESHOLD = 0.55;

const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);
cv::Scalar GREEN = cv::Scalar(0, 255, 0);

enum State {
    awake = 15, drowsy = 16, Look_Forward = 17, yelling = 18
};

class Detection {
private:
    std::unique_ptr<cv::dnn::Net> net;
    int detected_id = 0;
public:
    Detection() {
        // read model
        //yolov5n.onnx"
        this->net = std::make_unique<cv::dnn::Net>(cv::dnn::readNet("../model/11_21_yolov8n.onnx"));
        // set cuda
        this->net->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	    this->net->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    ~Detection() {
    }
    void draw_label(cv::Mat& input_image, std::string&& label, int&& left, int&& top)
    {
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE,
                                              THICKNESS, &baseLine);
        top = fmax(top, label_size.height);
        cv::Point tlc = cv::Point(left, top);
        cv::Point brc = cv::Point(left + label_size.width,
                                  (top + label_size.height + baseLine));
        cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
        cv::putText(input_image, label, cv::Point(left,
                    top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW,
                    THICKNESS);
    }


    /* resize webcam image to fit model input size */
    std::vector<cv::Mat> pre_process(cv::Mat&& input_image)
    {
        cv::Mat blob;
        cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH,
                               INPUT_HEIGHT), cv::Scalar(), true, false);
        this->net->setInput(blob);

        std::vector<cv::Mat> outputs;
        this->net->forward(outputs, this->net->getUnconnectedOutLayersNames());
        return outputs;
    }

    /* main process */
    cv::Mat post_process(cv::Mat&& input_image, std::vector<cv::Mat>& outputs,
                         const std::vector<std::string>& class_name)
    {

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        bool isYolov8 = false;

        int dimensions = outputs[0].size[2];
        int rows = outputs[0].size[1];;

        if (dimensions > rows) { // check is yolov8
            isYolov8 = true;
            rows = outputs[0].size[2];
            dimensions = outputs[0].size[1];

            outputs[0] = outputs[0].reshape(1, dimensions);
            cv::transpose(outputs[0], outputs[0]);
        }

        float x_factor = input_image.cols / INPUT_WIDTH;
        float y_factor = input_image.rows / INPUT_HEIGHT;
        float *data = (float *)outputs[0].data;

        this->detected_id = 0;

        for (int i = 0; i < rows; ++i) {
            if (!isYolov8) { // if yolov5
                float confidence = data[4];
                if (confidence >= CONFIDENCE_THRESHOLD) {
                    float* classes_scores = data + 5;
                    cv::Mat scores(1, class_name.size(), CV_32FC1,
                                   classes_scores);

                    cv::Point class_id;
                    double max_class_score;
                    cv::minMaxLoc(scores, 0, &max_class_score,
                                  0, &class_id);
                    if (max_class_score > SCORE_THRESHOLD) {
                        confidences.emplace_back(confidence);
                        class_ids.emplace_back(class_id.x);

                        float cx = data[0];
                        float cy = data[1];
                        float w = data[2];
                        float h = data[3];

                        int left = (int)((cx - 0.5 * w) * x_factor);
                        int top = (int)((cy - 0.5 * h) * y_factor);
                        int width = (int)(w * x_factor);
                        int height = (int)(h * y_factor);
                        boxes.emplace_back(cv::Rect(left, top,
                                           width, height));
                    }   
                }
            } else { // if yolov8
                float *classes_scores = data+4;
                cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double maxClassScore;
                minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

                if (maxClassScore > CONFIDENCE_THRESHOLD) {
                    confidences.emplace_back(maxClassScore);
                    class_ids.emplace_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                     boxes.emplace_back(cv::Rect(left, top, width, height));
                }
            }
            data += dimensions;
        }	
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD,
                          NMS_THRESHOLD, indices);
        for (int i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            int left = box.x;
            int top = box.y;
            int width = box.width;
            int height = box.height;

            cv::rectangle(input_image, cv::Point(left, top),
            cv::Point(left + width, top + height),
                    BLUE, 3 * THICKNESS);
            std::string label = cv::format("%.2f", confidences[idx]);
            label = class_name[class_ids[idx]] + ":" + label;
            draw_label(input_image, std::move(label), std::move(left), std::move(top));
            if (this->detected_id < class_ids[idx]) this->detected_id = class_ids[idx];
        }
        return input_image;
    }
    int getDetectedId() {
        return this->detected_id;
    }
};

/* looping audio file */
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    ma_decoder* pDecoder = (ma_decoder*)pDevice->pUserData;
    if (pDecoder == NULL) {
        return;
    }

    /* Reading PCM frames will loop based on what we specified when called ma_data_source_set_looping(). */
    ma_data_source_read_pcm_frames(pDecoder, pOutput, frameCount, NULL);

    (void)pInput;
}

bool check_file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

std::experimental::filesystem::path get_executable_path() {
#ifdef _WIN32
    wchar_t szPath[MAX_PATH];
    GetModuleFileNameW(nullptr, szPath, MAX_PATH);
    return std::experimental::filesystem::path{szPath}.parent_path();
#elif __APPLE__
    char szPath[PATH_MAX];
    uint32_t len = PATH_MAX;
    if (!_NSGetExecutablePath(szPath, &len))
        return std::experimental::filesystem::path{szPath}.parent_path();
    return "";
#else
    char szPath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", szPath, sizeof(szPath) - 1);
    if (len != -1) {
        szPath[len] = '\0';
        return std::experimental::filesystem::path{szPath}.parent_path();
    }
    return "";
#endif
}

int main(int argc, char* argv[])
{
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    std::experimental::filesystem::path sharePath = get_executable_path().parent_path();
    std::string modelPath = (argc == 2) ? argv[1] : sharePath.string() + "/model/yolo-v5n.onnx";

    if (!check_file_exists(modelPath)) {
        std::cerr << "Model: " << modelPath << " not exists." << std::endl;
        modelPath = sharePath.string() + "/model/yolo-v5n.onnx";
        if (!check_file_exists(modelPath)) {
            std::cerr << "Model: " << modelPath << " not exists." << std::endl;
            return -2;
        }
    }

    /* set audio file */
    ma_result result;
    ma_decoder decoder;
    ma_device_config deviceConfig;
    ma_device device;

    std::string soundPath = sharePath.string() + "/sound/sample.mp3";
    result = ma_decoder_init_file(soundPath.c_str(), NULL, &decoder);
    if (result != MA_SUCCESS) {
        std::cerr << "Sound: " << soundPath << " not exists." << std::endl;
        return -2;
    }
    
    /* set audio */
    ma_data_source_set_looping(&decoder, MA_TRUE);
    deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = decoder.outputFormat;
    deviceConfig.playback.channels = decoder.outputChannels;
    deviceConfig.sampleRate        = decoder.outputSampleRate;
    deviceConfig.dataCallback      = data_callback;
    deviceConfig.pUserData         = &decoder;


    Detection detection;
    std::vector<std::string> class_list{"dog","person","cat","tv","car","meatballs","marinara sauce","tomato soup","chicken noodle soup","french onion soup","chicken breast","ribs","pulled pork","hamburger","cavity","awake","drowsy","Look_Forward","yelling"};
    std::vector<cv::Mat> detections;
    cv::VideoCapture cap(0);
    cv::Mat frame;
    ma_device_init(NULL, &deviceConfig, &device);
    std::string perclos = "";


    std::map<std::string, int> detectedCount{{"awake", 0}, {"drowsy", 0}, {"Look_Forward", 0}, {"yelling", 0}};
    bool drowsy = false; // play mp3 file
    double drowsy_cnt = 0.0;
    int timer = 0; // reset timer, drowsy, result every 30 sec
    int awake_result = 0;
    int drowsy_result = 0;
    int look_forward_result = 0;
    int yelling_result = 0;
    cv::Scalar currentTextColor;

    /* result */
    while (1) {
        cap.read(frame);
        
        detections = detection.pre_process(std::move(frame));
        cv::Mat img = detection.post_process(std::move(frame),
                                             detections, class_list);
        if (!drowsy) {
            if (detection.getDetectedId() == static_cast<int>(State::drowsy)) {
                drowsy_cnt += 1.0;
                detectedCount["drowsy"] += 130;
            } else if (detection.getDetectedId() == static_cast<int>(State::Look_Forward)) {
                drowsy_cnt += 0.7;
                detectedCount["Look_Forward"] += 130;
            } else if (detection.getDetectedId() == static_cast<int>(State::yelling)) {
                drowsy_cnt += 2.0;
                detectedCount["yelling"] += 130;
            } else if (detection.getDetectedId() == static_cast<int>(State::awake)) {
                detectedCount["awake"] += 130;
            }

            if (drowsy_cnt < 81.7) {
                perclos = "Safe";
                currentTextColor = GREEN;
            } else if ((drowsy_cnt >= 81.7) && (drowsy_cnt < 138.0)) {
                perclos = "Drowsiness Suspiction";
                currentTextColor = YELLOW;
            } else if (drowsy_cnt >= 138.0) { // looping audio file
                perclos = "Drowsy Driving";
                currentTextColor = RED;
                drowsy = true;
                ma_device_start(&device);
                drowsy_cnt = 0.0;
                timer = 0;
                awake_result = detectedCount["awake"];
                look_forward_result = detectedCount["Look_Forward"];
                drowsy_result = detectedCount["drowsy"];
                yelling_result = detectedCount["yelling"];
                detectedCount["drowsy"] = 0;
                detectedCount["Look_Forward"] = 0;
                detectedCount["yelling"] = 0;
                detectedCount["awake"] = 0;
            }
        } else {
            if (detection.getDetectedId() == static_cast<int>(State::awake)) {
                drowsy_cnt += 1.5;
            }

            if (drowsy_cnt >= 77.0) { // stop audio file
                drowsy = false;
                ma_device_stop(&device);
                drowsy_cnt = 0.0;
                timer = 0;
                awake_result = detectedCount["awake"];
                look_forward_result = detectedCount["Look_Forward"];
                drowsy_result = detectedCount["drowsy"];
                yelling_result = detectedCount["yelling"];
                detectedCount["drowsy"] = 0;
                detectedCount["Look_Forward"] = 0;
                detectedCount["yelling"] = 0;
                detectedCount["awake"] = 0;
            }
        }

        timer += 1.0;
        if (timer >= 230.0) {
            drowsy_cnt = 0.0;
            timer = 0;

        }
        cv::putText(img, perclos, cv::Point(40, 50), cv::FONT_HERSHEY_COMPLEX, 2, std::move(currentTextColor), THICKNESS);
        cv::putText(img, "awake : " + std::to_string(awake_result) + "msec", cv::Point(40, 100), cv::FONT_HERSHEY_COMPLEX, FONT_SCALE, BLACK, THICKNESS);
        cv::putText(img, "look_forward : " + std::to_string(look_forward_result) + "msec", cv::Point(40, 150), cv::FONT_HERSHEY_COMPLEX, FONT_SCALE, BLACK, THICKNESS);
        cv::putText(img, "drowsy : " + std::to_string(drowsy_result) + "msec", cv::Point(40, 200), cv::FONT_HERSHEY_COMPLEX, FONT_SCALE, BLACK, THICKNESS);
        cv::putText(img, "yelling : " + std::to_string(yelling_result) + "msec", cv::Point(40, 250), cv::FONT_HERSHEY_COMPLEX, FONT_SCALE, BLACK, THICKNESS);
        cv::imshow("Output", std::move(img)); // show result
        if (cv::waitKey(27) >= 0) break;
    }
    ma_device_uninit(&device);
    ma_decoder_uninit(&decoder);
    return 0;
}
