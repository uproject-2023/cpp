#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>
#include <memory>

const float INPUT_WIDTH = 224.0;
const float INPUT_HEIGHT = 224.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

cv::Scalar BLACK = cv::Scalar(0, 0, 0);
cv::Scalar BLUE = cv::Scalar(255, 178, 50);
cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
cv::Scalar RED = cv::Scalar(0, 0, 255);

void draw_label(cv::Mat& input_image, std::string label, int left, int top)
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


/* resize webcam image to model input size */
std::vector<cv::Mat> pre_process(cv::Mat &input_image, cv::dnn::Net &net)
{
	cv::Mat resized_image;
        cv::resize(input_image, resized_image, cv::Size(224, 224));
	cv::Mat blob;
	cv::dnn::blobFromImage(resized_image, blob, 1.0, cv::Size(INPUT_WIDTH,
			  INPUT_HEIGHT), cv::Scalar(127.5, 127.5, 127.5), true, false);
	net.setInput(blob);

	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	return outputs;
}

cv::Mat post_process(cv::Mat&& input_image, std::vector<cv::Mat>& outputs,
		     const std::vector<std::string>& class_name)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	for(int i = 0; i < outputs.size(); i++) {
		float *data = (float *)outputs[i].data;
		for(int j = 0; j < outputs[i].total(); j += 7) {
			float confidence = data[j+2];
			int classId = (int)data[j+1];
			if(confidence > CONFIDENCE_THRESHOLD) {
				float left = data[j+3] * input_image.cols;
				float top = data[j+4] * input_image.rows;
				float right = data[j+5] * input_image.cols;
				float bottom = data[j+6] * input_image.rows;

				// Add 1 because cv::Rect() defines the boundary as left and top are inclusive,
				//  and as right and bottom are exclusive?
				float width = right - left + 1; 
				float height = bottom - top + 1;

				class_ids.push_back(classId - 1); // classID=0 is background, and we have to start
													// the index from 1 as 0 to get a corresponding
													// class name from the class list.
				confidences.push_back(confidence);
				boxes.push_back(cv::Rect(left, top, width, height));
			}            
		}
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
		draw_label(input_image, label, left, top);
	}
	return input_image;
}

int main()
{
	std::vector<std::string> class_list{"dog", "person"};
	cv::dnn::Net net;
	net = cv::dnn::readNetFromTensorflow("../model/frozen_inference_graph.pb", "../model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt");

	/* set backend to CUDA for GPU acceleration */
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	
	std::vector<cv::Mat> detections;
	cv::VideoCapture cap(0);
	cv::Mat frame;
	while (1) {
		cap.read(frame);
		detections = pre_process(frame, net);
		cv::Mat img = post_process(std::move(frame), detections,
					   class_list);

		std::vector<double> layersTimes;
		double freq = cv::getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = cv::format("Inference time : %.2f ms", t);
		cv::putText(img, label, cv::Point(20, 40), FONT_FACE,
			    FONT_SCALE, RED);
		cv::imshow("Output", std::move(img));
		if (cv::waitKey(27) >= 0) break;
	}
	return 0;
}
