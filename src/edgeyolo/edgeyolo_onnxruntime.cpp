#include "edgeyolo_onnxruntime.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dml_provider_factory.h>
#endif

namespace edgeyolo_cpp {

EdgeYOLOONNXRuntime::EdgeYOLOONNXRuntime(file_name_t path_to_model, int intra_op_num_threads,
					 int num_classes, int inter_op_num_threads,
					 const std::string &use_gpu_, int device_id,
					 bool use_parallel, float nms_th, float conf_th)
	: AbcEdgeYOLO(path_to_model, intra_op_num_threads, inter_op_num_threads, use_gpu_,
		      device_id, use_parallel, nms_th, conf_th, num_classes)
{
}

std::vector<Object> EdgeYOLOONNXRuntime::inference(const cv::Mat &frame)
{
	ONNXRuntimeModel::inference(frame, 0);

	if (this->output_buffer_.empty()) {
		obs_log(LOG_ERROR, "Output buffer is empty");
		return {};
	}

	float *net_pred = (float *)this->output_buffer_[0].get();
	if (net_pred == nullptr) {
		obs_log(LOG_ERROR, "Net prediction is null");
		return {};
	}

	if (this->num_array_ <= 0) {
		obs_log(LOG_ERROR, "Invalid num_array: %d", this->num_array_);
		return {};
	}

	float scale = std::min((float)input_w_[0] / (float)frame.cols,
			     (float)input_h_[0] / (float)frame.rows);
	std::vector<Object> objects;
	decode_outputs(net_pred, this->num_array_, objects, this->bbox_conf_thresh_, scale,
		       frame.cols, frame.rows);
	return objects;
}

} // namespace edgeyolo_cpp
