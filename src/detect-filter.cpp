#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <nlohmann/json.hpp>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "ort-model/utils.hpp"
#include "detect-filter-utils.h"
#include "edgeyolo/edgeyolo_onnxruntime.hpp"
#include "yunet/YuNet.h"

#define EXTERNAL_MODEL_SIZE "!!!EXTERNAL_MODEL!!!"
#define FACE_DETECT_MODEL_SIZE "!!!FACE_DETECT!!!"

struct detect_filter : public filter_data {};

// 异步推理线程函数声明
void inference_worker(struct detect_filter *tf);

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Detect");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop,
			    const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}

static bool enable_advanced_settings(obs_properties_t *ppts, obs_property_t *p,
				     obs_data_t *settings)
{
	const bool enabled = obs_data_get_bool(settings, "advanced");

	for (const char *prop_name :
	     {"threshold", "useGPU", "numThreads", "model_size", "detected_object",
	      "save_detections_path", "crop_group", "min_size_threshold"}) {
		p = obs_properties_get(ppts, prop_name);
		obs_property_set_visible(p, enabled);
	}

	return true;
}

void set_class_names_on_object_category(obs_property_t *object_category,
					std::vector<std::string> class_names)
{
	std::vector<std::pair<size_t, std::string>> indexed_classes;
	for (size_t i = 0; i < class_names.size(); ++i) {
		const std::string &class_name = class_names[i];
		std::string class_name_cap = class_name;
		class_name_cap[0] = (char)std::toupper((int)class_name_cap[0]);
		indexed_classes.push_back({i, class_name_cap});
	}

	std::sort(indexed_classes.begin(), indexed_classes.end(),
		  [](const std::pair<size_t, std::string> &a,
		     const std::pair<size_t, std::string> &b) { return a.second < b.second; });

	obs_property_list_clear(object_category);

	obs_property_list_add_int(object_category, obs_module_text("All"), -1);

	for (const auto &indexed_class : indexed_classes) {
		obs_property_list_add_int(object_category, indexed_class.second.c_str(),
					  (int)indexed_class.first);
	}
}

void read_model_config_json_and_set_class_names(const char *model_file, obs_properties_t *props_,
						obs_data_t *settings, struct detect_filter *tf_)
{
	if (model_file == nullptr || model_file[0] == '\0' || strlen(model_file) == 0) {
		obs_log(LOG_ERROR, "Model file path is empty");
		return;
	}

	std::string json_file = model_file;
	json_file.replace(json_file.find(".onnx"), 5, ".json");
	std::ifstream file(json_file);
	if (!file.is_open()) {
		obs_data_set_string(settings, "error", "JSON file not found");
		obs_log(LOG_ERROR, "JSON file not found: %s", json_file.c_str());
	} else {
		obs_data_set_string(settings, "error", "");
		nlohmann::json j;
		file >> j;
		if (j.contains("names")) {
			std::vector<std::string> labels = j["names"];
			set_class_names_on_object_category(
				obs_properties_get(props_, "object_category"), labels);
			tf_->classNames = labels;
		} else {
			obs_data_set_string(settings, "error",
					    "JSON file does not contain 'names' field");
			obs_log(LOG_ERROR, "JSON file does not contain 'names' field");
		}
	}
}

obs_properties_t *detect_filter_properties(void *data)
{
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	obs_properties_t *props = obs_properties_create();

	obs_properties_add_bool(props, "inference_enabled", obs_module_text("ToggleInference"));

	obs_properties_add_bool(props, "preview", obs_module_text("Preview"));

	obs_property_t *object_category =
		obs_properties_add_list(props, "object_category", obs_module_text("ObjectCategory"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	set_class_names_on_object_category(object_category, edgeyolo_cpp::COCO_CLASSES);
	tf->classNames = edgeyolo_cpp::COCO_CLASSES;

	obs_property_t *advanced =
		obs_properties_add_bool(props, "advanced", obs_module_text("Advanced"));

	obs_property_set_modified_callback(advanced, enable_advanced_settings);

	obs_properties_t *crop_group_props = obs_properties_create();
	obs_property_t *crop_group =
		obs_properties_add_group(props, "crop_group", obs_module_text("CropGroup"),
					 OBS_GROUP_CHECKABLE, crop_group_props);

	obs_property_set_modified_callback(crop_group, [](obs_properties_t *props_,
							  obs_property_t *, obs_data_t *settings) {
		const bool enabled = obs_data_get_bool(settings, "crop_group");
		for (auto prop_name : {"crop_left", "crop_right", "crop_top", "crop_bottom"}) {
			obs_property_t *prop = obs_properties_get(props_, prop_name);
			obs_property_set_visible(prop, enabled);
		}
		return true;
	});

	obs_properties_add_int_slider(crop_group_props, "crop_left", obs_module_text("CropLeft"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_right", obs_module_text("CropRight"),
				      0, 1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_top", obs_module_text("CropTop"), 0,
				      1000, 1);
	obs_properties_add_int_slider(crop_group_props, "crop_bottom",
				      obs_module_text("CropBottom"), 0, 1000, 1);

	obs_property_t *detected_obj_prop = obs_properties_add_text(
		props, "detected_object", obs_module_text("DetectedObject"), OBS_TEXT_DEFAULT);
	obs_property_set_enabled(detected_obj_prop, false);

	obs_properties_add_float_slider(props, "threshold", obs_module_text("ConfThreshold"), 0.0,
					1.0, 0.025);

	obs_properties_add_int_slider(props, "min_size_threshold",
				      obs_module_text("MinSizeThreshold"), 0, 10000, 1);

	obs_properties_add_path(props, "save_detections_path",
				obs_module_text("SaveDetectionsPath"), OBS_PATH_FILE_SAVE,
				"JSON file (*.json);;All files (*.*)", nullptr);

	obs_property_t *p_use_gpu =
		obs_properties_add_list(props, "useGPU", obs_module_text("InferenceDevice"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#if defined(__linux__) && defined(__x86_64__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUTensorRT"), USEGPU_TENSORRT);
#endif
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUDirectML"), USEGPU_DML);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	obs_properties_add_int_slider(props, "numThreads", obs_module_text("NumThreads"), 0, 8, 1);

	obs_property_t *model_size =
		obs_properties_add_list(props, "model_size", obs_module_text("ModelSize"),
					OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(model_size, obs_module_text("SmallFast"), "small");
	obs_property_list_add_string(model_size, obs_module_text("Medium"), "medium");
	obs_property_list_add_string(model_size, obs_module_text("LargeSlow"), "large");
	obs_property_list_add_string(model_size, obs_module_text("FaceDetect"),
				     FACE_DETECT_MODEL_SIZE);
	obs_property_list_add_string(model_size, obs_module_text("ExternalModel"),
				     EXTERNAL_MODEL_SIZE);

	obs_properties_add_path(props, "external_model_file", obs_module_text("ModelPath"),
				OBS_PATH_FILE, "ONNX files (*.onnx);;all files (*.*)",
				nullptr);

	obs_property_set_modified_callback2(
		model_size,
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			std::string model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = model_size_value == EXTERNAL_MODEL_SIZE;
			obs_property_t *prop = obs_properties_get(props_, "external_model_file");
			obs_property_set_visible(prop, is_external);
			if (!is_external) {
				if (model_size_value == FACE_DETECT_MODEL_SIZE) {
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						yunet::FACE_CLASSES);
					tf_->classNames = yunet::FACE_CLASSES;
				} else {
					set_class_names_on_object_category(
						obs_properties_get(props_, "object_category"),
						edgeyolo_cpp::COCO_CLASSES);
					tf_->classNames = edgeyolo_cpp::COCO_CLASSES;
				}
			} else {
				const char *model_file =
					obs_data_get_string(settings, "external_model_file");
				read_model_config_json_and_set_class_names(model_file, props_,
									   settings, tf_);
			}
			return true;
		},
		tf);

	obs_property_set_modified_callback2(
		obs_properties_get(props, "external_model_file"),
		[](void *data_, obs_properties_t *props_, obs_property_t *p, obs_data_t *settings) {
			UNUSED_PARAMETER(p);
			const char *model_size_value = obs_data_get_string(settings, "model_size");
			bool is_external = strcmp(model_size_value, EXTERNAL_MODEL_SIZE) == 0;
			if (!is_external) {
				return true;
			}
			struct detect_filter *tf_ = reinterpret_cast<detect_filter *>(data_);
			const char *model_file =
				obs_data_get_string(settings, "external_model_file");
			read_model_config_json_and_set_class_names(model_file, props_, settings,
								   tf_);
			return true;
		},
		tf);

	std::string basic_info =
		std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "inference_enabled", false);
	obs_data_set_default_bool(settings, "advanced", false);
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", true);
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_string(settings, "model_size", "small");
	obs_data_set_default_int(settings, "object_category", -1);
	obs_data_set_default_string(settings, "save_detections_path", "");
	obs_data_set_default_bool(settings, "crop_group", false);
	obs_data_set_default_int(settings, "crop_left", 0);
	obs_data_set_default_int(settings, "crop_right", 0);
	obs_data_set_default_int(settings, "crop_top", 0);
	obs_data_set_default_int(settings, "crop_bottom", 0);
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter update");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (!tf) {
		obs_log(LOG_ERROR, "Filter data is null");
		return;
	}

	// 在更新设置前临时禁用推理以避免并发问题
	bool was_inference_enabled = tf->inferenceEnabled;
	tf->inferenceEnabled = false; // 先停止推理
	tf->isDisabled = true;

	bool new_inference_enabled = obs_data_get_bool(settings, "inference_enabled");
	if (new_inference_enabled != was_inference_enabled) {
		obs_log(LOG_INFO, "Inference %s", new_inference_enabled ? "enabled" : "disabled");
	}

	tf->preview = obs_data_get_bool(settings, "preview");
	tf->conf_threshold = (float)obs_data_get_double(settings, "threshold");
	tf->objectCategory = (int)obs_data_get_int(settings, "object_category");
	tf->saveDetectionsPath = obs_data_get_string(settings, "save_detections_path");
	tf->crop_enabled = obs_data_get_bool(settings, "crop_group");
	tf->crop_left = (int)obs_data_get_int(settings, "crop_left");
	tf->crop_right = (int)obs_data_get_int(settings, "crop_right");
	tf->crop_top = (int)obs_data_get_int(settings, "crop_top");
	tf->crop_bottom = (int)obs_data_get_int(settings, "crop_bottom");
	tf->minAreaThreshold = (int)obs_data_get_int(settings, "min_size_threshold");

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");
	const std::string newModelSize = obs_data_get_string(settings, "model_size");

	bool reinitialize = false;
	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads ||
	    tf->modelSize != newModelSize) {
		obs_log(LOG_INFO, "Reinitializing model");
		reinitialize = true;

		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = nullptr;
		if (newModelSize == "small") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_256x416.onnx");
		} else if (newModelSize == "medium") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_480x800.onnx");
		} else if (newModelSize == "large") {
			modelFilepath_rawPtr =
				obs_module_file("models/edgeyolo_tiny_lrelu_coco_736x1280.onnx");
		} else if (newModelSize == FACE_DETECT_MODEL_SIZE) {
			modelFilepath_rawPtr =
				obs_module_file("models/face_detection_yunet_2023mar.onnx");
		} else if (newModelSize == EXTERNAL_MODEL_SIZE) {
			const char *external_model_file =
				obs_data_get_string(settings, "external_model_file");
			if (external_model_file == nullptr || external_model_file[0] == '\0' ||
			    strlen(external_model_file) == 0) {
				obs_log(LOG_ERROR, "External model file path is empty");
				tf->isDisabled = true;
				return;
			}
			modelFilepath_rawPtr = bstrdup(external_model_file);
		} else {
			obs_log(LOG_ERROR, "Invalid model size: %s", newModelSize.c_str());
			tf->isDisabled = true;
			return;
		}

		if (modelFilepath_rawPtr == nullptr) {
			obs_log(LOG_ERROR, "Unable to get model filename from plugin.");
			tf->isDisabled = true;
			return;
		}

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr,
						    -1, nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED, modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif
		bfree(modelFilepath_rawPtr);

		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;
		tf->modelSize = newModelSize;

		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = true;
		float nms_th_ = 0.45f;
		int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();
		tf->classNames = edgeyolo_cpp::COCO_CLASSES;

		if (tf->modelSize == EXTERNAL_MODEL_SIZE) {
#ifdef _WIN32
			std::wstring labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(L".onnx"), 5, L".json");
#else
			std::string labelsFilepath = tf->modelFilepath;
			labelsFilepath.replace(labelsFilepath.find(".onnx"), 5, ".json");
#endif
			std::ifstream labelsFile(labelsFilepath);
			if (labelsFile.is_open()) {
				nlohmann::json j;
				labelsFile >> j;
				if (j.contains("names")) {
					std::vector<std::string> labels = j["names"];
					num_classes_ = (int)labels.size();
					tf->classNames = labels;
				} else {
					obs_log(LOG_ERROR,
						"JSON file does not contain 'labels' field");
					tf->isDisabled = true;
					tf->onnxruntimemodel.reset();
					return;
				}
			} else {
				obs_log(LOG_ERROR, "Failed to open JSON file: %s",
					labelsFilepath.c_str());
				tf->isDisabled = true;
				tf->onnxruntimemodel.reset();
				return;
			}
		} else if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
			num_classes_ = 1;
			tf->classNames = yunet::FACE_CLASSES;
		}

		try {
			// 确保在重置模型时没有其他线程在使用它
			if (tf->onnxruntimemodel) {
				tf->onnxruntimemodel.reset();
			}
			if (tf->modelSize == FACE_DETECT_MODEL_SIZE) {
				tf->onnxruntimemodel = std::make_unique<yunet::YuNetONNX>(
					tf->modelFilepath, tf->numThreads, 50, tf->numThreads,
					tf->useGPU, onnxruntime_device_id_,
					onnxruntime_use_parallel_, nms_th_, tf->conf_threshold);
			} else {
				tf->onnxruntimemodel =
					std::make_unique<edgeyolo_cpp::EdgeYOLOONNXRuntime>(
						tf->modelFilepath, tf->numThreads, num_classes_,
						tf->numThreads, tf->useGPU, onnxruntime_device_id_,
						onnxruntime_use_parallel_, nms_th_,
						tf->conf_threshold);
			}
			obs_data_set_string(settings, "error", "");
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s", e.what());
			tf->isDisabled = true;
			tf->onnxruntimemodel.reset();
			return;
		}
	} else {
		obs_log(LOG_INFO, "Model already loaded, skipping reinitialization");
		tf->isDisabled = false;
	}

	if (tf->onnxruntimemodel) {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
	}

	if (reinitialize) {
		obs_log(LOG_INFO, "Detect Filter Options:");
		obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
		obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
		obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
		obs_log(LOG_INFO, "  Model Size: %s", tf->modelSize.c_str());
		obs_log(LOG_INFO, "  Preview: %s", tf->preview ? "true" : "false");
		obs_log(LOG_INFO, "  Threshold: %.2f", tf->conf_threshold);
		obs_log(LOG_INFO, "  Object Category: %s",
			obs_data_get_string(settings, "object_category"));
		obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");
#ifdef _WIN32
		obs_log(LOG_INFO, "  Model file path: %ls", tf->modelFilepath.c_str());
#else
		obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif
	}

	if (!reinitialize) {
		tf->isDisabled = false;
	}
	
	// 恢复推理启用状态 - 这个变量在tick函数中会被读取，所以不需要特殊保护
	// 因为简单的布尔赋值是原子操作
	tf->inferenceEnabled = new_inference_enabled;
	}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	if (tf) {
		tf->isDisabled = false;
	}
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	if (tf) {
		tf->isDisabled = true;
	}
}

/**                   FILTER CORE                     */

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	// 初始化所有成员变量
	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->stagesurface = nullptr;
	tf->lastDetectedObjectId = -1;
	tf->last_inference_time = std::chrono::steady_clock::time_point();
	tf->inferenceEnabled = false;
	tf->preview = true;
	tf->conf_threshold = 0.5f;
	tf->objectCategory = -1;
	tf->saveDetectionsPath = "";
	tf->crop_enabled = false;
	tf->crop_left = 0;
	tf->crop_right = 0;
	tf->crop_top = 0;
	tf->crop_bottom = 0;
	tf->minAreaThreshold = 0;
	tf->useGPU = "CPU";
	tf->numThreads = 1;
	tf->modelSize = "small";
	tf->isDisabled = false;
	tf->onnxruntimemodel = nullptr;
	tf->should_stop = false;
	tf->thread_running = false;

#if _WIN32
	tf->modelFilepath = L"";
#else
	tf->modelFilepath = "";
#endif

	detect_filter_update(tf, settings);

	// 启动推理线程
	try {
		tf->inference_thread = std::thread(inference_worker, tf);
		obs_log(LOG_INFO, "Inference thread started successfully");
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "Failed to start inference thread: %s", e.what());
	}

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;
		tf->should_stop = true;  // 设置停止标志

		// 唤醒推理线程（以防它正在等待）
		{
			std::lock_guard<std::mutex> lock(tf->queue_mutex);
			tf->queue_condition.notify_all();
		}

		// 等待推理线程结束
		if (tf->inference_thread.joinable()) {
			try {
				tf->inference_thread.join();
				obs_log(LOG_INFO, "Inference thread joined successfully");
			} catch (const std::exception &e) {
				obs_log(LOG_ERROR, "Error joining inference thread: %s", e.what());
			}
		}

		// 等待模型不再被使用后再清理
		{
			std::unique_lock<std::mutex> lock(tf->modelMutex);
			tf->onnxruntimemodel.reset();  // 显式重置模型
		}

		obs_enter_graphics();
		if (tf->texrender) {
			gs_texrender_destroy(tf->texrender);
		}
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		obs_leave_graphics();

		tf->~detect_filter();
		bfree(tf);
	}
}

// 异步推理线程函数
void inference_worker(struct detect_filter *tf)
{
	obs_log(LOG_INFO, "Starting inference worker thread");
	tf->thread_running = true;
	
	while (!tf->should_stop) {
		cv::Mat frame;
		{
			std::unique_lock<std::mutex> lock(tf->queue_mutex);
			if (tf->frame_queue.empty()) {
				// 等待新帧或停止信号
				tf->queue_condition.wait(lock, [&tf] { 
					return !tf->frame_queue.empty() || tf->should_stop.load(); 
				});
				
				if (tf->should_stop && tf->frame_queue.empty()) {
					break;
				}
			}
			
			if (!tf->frame_queue.empty()) {
				frame = tf->frame_queue.front();
				tf->frame_queue.pop();
			}
		}
		
		if (!frame.empty()) {
			// 执行推理
			std::vector<Object> objects;
			
			{
				// 使用模型进行推理
				std::unique_lock<std::mutex> lock(tf->modelMutex, std::try_to_lock);
				if (lock.owns_lock() && tf->onnxruntimemodel) {
					try {
						cv::Mat inferenceFrame;
						cv::Rect cropRect(0, 0, frame.cols, frame.rows);
						if (tf->crop_enabled) {
							cropRect = cv::Rect(tf->crop_left, tf->crop_top,
									    frame.cols - tf->crop_left - tf->crop_right,
									    frame.rows - tf->crop_top - tf->crop_bottom);
							cv::cvtColor(frame(cropRect), inferenceFrame, cv::COLOR_BGRA2BGR);
						} else {
							cv::cvtColor(frame, inferenceFrame, cv::COLOR_BGRA2BGR);
						}

						// 设置置信度阈值
						tf->onnxruntimemodel->setBBoxConfThresh(tf->conf_threshold);
						objects = tf->onnxruntimemodel->inference(inferenceFrame);
						tf->last_inference_time = std::chrono::steady_clock::now();

						obs_log(LOG_INFO, "Inference returned %d objects (before filtering)", objects.size());

						if (tf->crop_enabled) {
							for (Object &obj : objects) {
								obj.rect.x += (float)cropRect.x;
								obj.rect.y += (float)cropRect.y;
							}
						}

						if (tf->objectCategory != -1) {
							std::vector<Object> filtered_objects;
							for (const Object &obj : objects) {
								if (obj.label == tf->objectCategory) {
									filtered_objects.push_back(obj);
								}
							}
							objects = filtered_objects;
							obs_log(LOG_INFO, "After category filter: %d objects", objects.size());
						}

						if (!tf->saveDetectionsPath.empty()) {
							std::ofstream detectionsFile(tf->saveDetectionsPath);
							if (detectionsFile.is_open()) {
								nlohmann::json j;
								for (const Object &obj : objects) {
									nlohmann::json obj_json;
									obj_json["label"] = obj.label;
									obj_json["confidence"] = obj.prob;
									obj_json["rect"] = {{"x", obj.rect.x},
											    {"y", obj.rect.y},
											    {"width", obj.rect.width},
											    {"height", obj.rect.height}};
									obj_json["id"] = obj.id;
									j.push_back(obj_json);
								}
								detectionsFile << j.dump(4);
								detectionsFile.close();
							}
						}
					} catch (const std::exception &e) {
						obs_log(LOG_ERROR, "Inference error: %s", e.what());
					}
				}
			}
			
			// 处理检测结果
			if (objects.size() > 0 && tf->lastDetectedObjectId != objects[0].label) {
				tf->lastDetectedObjectId = objects[0].label;
				obs_source_t *source = tf->source;
				if (source) {
					obs_data_t *source_settings = obs_source_get_settings(source);
					if (source_settings) {
						obs_data_set_string(source_settings, "detected_object",
								    tf->classNames[objects[0].label].c_str());
						obs_data_release(source_settings);
					}
				}
			} else if (objects.empty() && tf->lastDetectedObjectId != -1) {
				tf->lastDetectedObjectId = -1;
				obs_source_t *source = tf->source;
				if (source) {
					obs_data_t *source_settings = obs_source_get_settings(source);
					if (source_settings) {
						obs_data_set_string(source_settings, "detected_object", "");
						obs_data_release(source_settings);
					}
				}
			}
			
			// 绘制检测结果
			if (tf->preview) {
				cv::Mat result_frame = frame.clone();
				cv::Mat draw_frame;
				cv::cvtColor(result_frame, draw_frame, cv::COLOR_BGRA2BGR);

				cv::Rect cropRect(0, 0, frame.cols, frame.rows);
				if (tf->crop_enabled) {
					cropRect = cv::Rect(tf->crop_left, tf->crop_top,
							    frame.cols - tf->crop_left - tf->crop_right,
							    frame.rows - tf->crop_top - tf->crop_bottom);
					drawDashedRectangle(draw_frame, cropRect, cv::Scalar(0, 255, 0), 5, 8, 15);
				}
				
				if (objects.size() > 0) {
					draw_objects(draw_frame, objects, tf->classNames);
					obs_log(LOG_INFO, "Drew %d boxes on frame", objects.size());
				}

				{
					std::lock_guard<std::mutex> lock(tf->outputLock);
					cv::cvtColor(draw_frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
				}
			}
		}
	}
	
	obs_log(LOG_INFO, "Stopping inference worker thread");
	tf->thread_running = false;
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (!tf) {
		return;
	}

	static bool last_inference_state = false;
	if (last_inference_state != tf->inferenceEnabled) {
		last_inference_state = tf->inferenceEnabled;
		obs_log(LOG_INFO, "Inference state changed to: %s", tf->inferenceEnabled ? "ENABLED" : "DISABLED");
	}

	if (tf->isDisabled) {
		obs_log(LOG_WARNING, "Filter is disabled, skipping tick");
		return;
	}

	if (!tf->source || !obs_source_enabled(tf->source)) {
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf, width, height)) {
		return;
	}

	cv::Mat imageBGRA;
	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		if (tf->inputBGRA.empty()) {
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	if (!tf->onnxruntimemodel) {
		obs_log(LOG_WARNING, "Model not loaded, showing original image");
		if (tf->preview) {
			std::lock_guard<std::mutex> lock(tf->outputLock);
			tf->outputPreviewBGRA = imageBGRA.clone();
		}
		return;
	}

	// 将帧添加到推理队列（仅当推理启用且队列未满时）
	if (tf->inferenceEnabled) {
		auto now = std::chrono::steady_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			now - tf->last_inference_time).count();
		
		if (elapsed_ms >= tf->MIN_INFERENCE_INTERVAL_MS) {
			// 检查队列大小，避免无限增长
			{
				std::lock_guard<std::mutex> lock(tf->queue_mutex);
				// 只保留最新的一帧，丢弃旧帧以避免延迟
				while (tf->frame_queue.size() > 1) {
					tf->frame_queue.pop();
				}
				// 添加新帧到队列
				tf->frame_queue.push(imageBGRA.clone());
			}
			// 通知推理线程有新帧可用
			tf->queue_condition.notify_one();
		}
	}
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (!tf) {
		obs_source_skip_video_filter(nullptr);
		return;
	}

	if (!tf->preview) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target) {
		obs_source_skip_video_filter(tf->source);
		return;
	}
	uint32_t width = obs_source_get_base_width(target);
	uint32_t height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// 获取预览输出或原始输入
	cv::Mat outputBGRA;
	{
		std::lock_guard<std::mutex> lock(tf->outputLock);
		if (!tf->outputPreviewBGRA.empty() &&
		    (uint32_t)tf->outputPreviewBGRA.cols == width &&
		    (uint32_t)tf->outputPreviewBGRA.rows == height) {
			outputBGRA = tf->outputPreviewBGRA.clone(); // 使用处理后的图像（带检测框）
		} else {
			// 如果没有预览输出，则尝试使用输入图像
			std::lock_guard<std::mutex> lock_input(tf->inputBGRALock);
			if (!tf->inputBGRA.empty()) {
				outputBGRA = tf->inputBGRA.clone();
			}
		}
	}

	// 如果仍然没有图像数据，则跳过过滤器
	if (outputBGRA.empty()) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// 转换颜色空间并绘制图形
	cv::Mat frameBGR;
	cv::cvtColor(outputBGRA, frameBGR, cv::COLOR_BGRA2BGR);

	// 始终绘制十字和圆圈
	int center_x = frameBGR.cols / 2;
	int center_y = frameBGR.rows / 2;

	cv::Scalar cross_color = cv::Scalar(0, 255, 0);
	cv::Scalar circle_color = cv::Scalar(0, 0, 255);

	cv::line(frameBGR, cv::Point(center_x - 30, center_y), 
		 cv::Point(center_x + 30, center_y), cross_color, 2);
	cv::line(frameBGR, cv::Point(center_x, center_y - 30), 
		 cv::Point(center_x, center_y + 30), cross_color, 2);

	int circle_radius = 50;
	cv::circle(frameBGR, cv::Point(center_x, center_y), circle_radius, circle_color, 2);

	// 创建一个新的矩阵来存储最终结果，避免修改原始数据
	cv::Mat finalOutputBGRA;
	cv::cvtColor(frameBGR, finalOutputBGRA, cv::COLOR_BGR2BGRA);

	if (!finalOutputBGRA.empty() && finalOutputBGRA.data) {
		// 确保数据大小正确
		size_t expected_size = width * height * 4; // 4 bytes per pixel for BGRA
		if (finalOutputBGRA.total() * finalOutputBGRA.elemSize() >= expected_size) {
			gs_texture_t *tex = gs_texture_create(width, height, GS_BGRA, 1,
							      (const uint8_t **)&finalOutputBGRA.data, 0);
			if (tex) {
				gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
				gs_technique_t *tech = gs_effect_get_technique(effect, "Draw");
				gs_eparam_t *image_param = gs_effect_get_param_by_name(effect, "image");

				gs_effect_set_texture(image_param, tex);

				gs_technique_begin(tech);
				gs_technique_begin_pass(tech, 0);
				gs_draw_sprite(tex, 0, 0, 0);
				gs_technique_end_pass(tech);
				gs_technique_end(tech);

				gs_texture_destroy(tex);
			}
		}
	}
}