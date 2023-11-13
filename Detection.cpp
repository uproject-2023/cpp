#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <iostream>
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"


const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.3;
const float NMS_THRESHOLD = 0.25;
const float CONFIDENCE_THRESHOLD = 0.25;

const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);


class Detection {
private:
    std::unique_ptr<cv::dnn::Net> net;
    bool drowsy_detected = false;
public:
    Detection() {
        // read model
        this->net = std::make_unique<cv::dnn::Net>(cv::dnn::readNet("../model/last.onnx"));
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
    cv::Mat post_process(cv::Mat&& input_image, const std::vector<cv::Mat>& outputs,
                         const std::vector<std::string>& class_name)
    {
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float x_factor = input_image.cols / INPUT_WIDTH;
        float y_factor = input_image.rows / INPUT_HEIGHT;
        float *data = (float *)outputs[0].data;
        const int dimensions = 22;
        const int rows = 25200;

        for (int i = 0; i < rows; ++i) {
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
            data += 22;
        }	
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD,
                          NMS_THRESHOLD, indices);
        bool drowsyDetected = false;
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
            if (class_ids[idx] >= 16) drowsyDetected = true;
        }
        this->drowsy_detected = drowsyDetected;
        return input_image;
    }

    /* return fps */
    double getFPS() {
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        return this->net->getPerfProfile(layersTimes) / freq;
    }

    bool isDrowsyDetected() {
        return this->drowsy_detected;
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

int main()
{
    /* set audio */
    ma_result result;
    ma_decoder decoder;
    ma_device_config deviceConfig;
    ma_device device;
    result = ma_decoder_init_file("../sound/sample.mp3", NULL, &decoder);
    if (result != MA_SUCCESS) {
        return -2;
    }
    
    ma_data_source_set_looping(&decoder, MA_TRUE);
    deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = decoder.outputFormat;
    deviceConfig.playback.channels = decoder.outputChannels;
    deviceConfig.sampleRate        = decoder.outputSampleRate;
    deviceConfig.dataCallback      = data_callback;
    deviceConfig.pUserData         = &decoder;


    Detection detection;
    std::vector<std::string> class_list{"dog","person","cat","tv","car","meatballs","marinara sauce","tomato soup","chicken noodle soup","french onion soup","chicken breast","ribs","pulled pork","hamburger","cavity","awake","drowsy"};
    std::vector<cv::Mat> detections;
    cv::VideoCapture cap(0);
    cv::Mat frame;
    ma_device_init(NULL, &deviceConfig, &device);

    bool drowsy = false;
    bool playing = false;
    int drowsy_cnt = 0;

    while (1) {
        cap.read(frame);
        detections = detection.pre_process(std::move(frame));
        cv::Mat img = detection.post_process(std::move(frame),
                                             detections, class_list);
        if (detection.isDrowsyDetected()) { // start counting when drowsy detected
            if (drowsy) drowsy_cnt++;
            else drowsy = true;
        } else { // reset when awake
            drowsy = false;
            drowsy_cnt = 0;
            playing = false;
            ma_device_stop(&device);
        }

        if (drowsy_cnt > 10) { // looping audio file
            if (!playing) {
                playing = true;
                ma_device_start(&device);
            }
        }

        double t = detection.getFPS();
        std::string label = cv::format("Inference time : %.2f ms", t);
        cv::putText(img, label, cv::Point(20, 40), FONT_FACE,
                    FONT_SCALE, RED);
        cv::imshow("Output", std::move(img));
        if (cv::waitKey(27) >= 0) break;
    }

    ma_device_uninit(&device);
    ma_decoder_uninit(&decoder);
    return 0;
}
