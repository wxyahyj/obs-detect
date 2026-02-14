#ifndef ONNXRUNTIME_MODEL_H
#define ONNXRUNTIME_MODEL_H

#include &lt;opencv2/core/types.hpp&gt;
#include &lt;onnxruntime_cxx_api.h&gt;

#include &lt;vector&gt;
#include &lt;array&gt;
#include &lt;string&gt;
#include &lt;tuple&gt;

#include "types.hpp"

class ONNXRuntimeModel {
public:
	ONNXRuntimeModel(file_name_t path_to_model, int intra_op_num_threads, int num_classes,
			 int inter_op_num_threads = 1, const std::string &amp;use_gpu_ = "",
			 int device_id = 0, bool use_parallel = false, float nms_th = 0.45f,
			 float conf_th = 0.3f);
	virtual ~ONNXRuntimeModel() {
		input_tensor_.clear();
		output_tensor_.clear();
		input_buffer_.clear();
		output_buffer_.clear();
	}

	void setBBoxConfThresh(float thresh) { this-&gt;bbox_conf_thresh_ = thresh; }
	void setNmsThresh(float thresh) { this-&gt;nms_thresh_ = thresh; }

	virtual std::vector&lt;Object&gt; inference(const cv::Mat &amp;frame) = 0;

protected:
	cv::Mat static_resize(const cv::Mat &amp;img, const int input_index);
	void blobFromImage(const cv::Mat &amp;img, float *blob_data);
	float intersection_area(const Object &amp;a, const Object &amp;b);
	void qsort_descent_inplace(std::vector&lt;Object&gt; &amp;faceobjects, int left, int right);
	void qsort_descent_inplace(std::vector&lt;Object&gt; &amp;objects);
	void nms_sorted_bboxes(const std::vector&lt;Object&gt; &amp;objects, std::vector&lt;int&gt; &amp;picked,
			       const float nms_threshold);

	void inference(const cv::Mat &amp;frame, const int input_index);

	std::vector&lt;int&gt; input_w_;
	std::vector&lt;int&gt; input_h_;
	float nms_thresh_;
	float bbox_conf_thresh_;
	int num_classes_;
	bool use_parallel_;
	int inter_op_num_threads_;
	int intra_op_num_threads_;
	int device_id_;
	std::string use_gpu;

	Ort::Session session_{nullptr};
	Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Default"};

	std::vector&lt;Ort::Value&gt; input_tensor_;
	std::vector&lt;Ort::Value&gt; output_tensor_;
	std::vector&lt;std::string&gt; input_name_;
	std::vector&lt;std::string&gt; output_name_;
	std::vector&lt;std::unique_ptr&lt;uint8_t[]&gt;&gt; input_buffer_;
	std::vector&lt;std::unique_ptr&lt;uint8_t[]&gt;&gt; output_buffer_;
	std::vector&lt;Ort::ShapeInferContext::Ints&gt; output_shapes_;
};

#endif
