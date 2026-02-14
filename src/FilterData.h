#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include <opencv2/core.hpp>
#include <mutex>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <queue>
#include "ort-model/ONNXRuntimeModel.h"

struct filter_data {
	std::string useGPU;
	uint32_t numThreads;
	float conf_threshold;
	std::string modelSize;

	int minAreaThreshold;
	int objectCategory;
	int lastDetectedObjectId;
	std::string saveDetectionsPath;
	bool crop_enabled;
	int crop_left;
	int crop_right;
	int crop_top;
	int crop_bottom;

	obs_source_t *source;
	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;

	cv::Mat inputBGRA;
	cv::Mat outputPreviewBGRA;

	bool isDisabled;
	bool preview;

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;

	std::unique_ptr<ONNXRuntimeModel> onnxruntimemodel;
	std::vector<std::string> classNames;

	std::chrono::steady_clock::time_point last_inference_time;
	static constexpr int MIN_INFERENCE_INTERVAL_MS = 100;

	std::thread inference_thread;
	std::atomic<bool> inference_thread_running;
	std::queue<cv::Mat> frame_queue;
	std::mutex queue_mutex;
	std::condition_variable queue_cv;
	std::vector<Object> latest_detections;
	std::mutex detections_mutex;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */
