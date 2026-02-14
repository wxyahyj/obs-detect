#include "ONNXRuntimeModel.h"

#ifdef _WIN32
#include &lt;dml_provider_factory.h&gt;
#endif

#include "plugin-support.h"

#include &lt;obs.h&gt;
#include &lt;stdexcept&gt;

ONNXRuntimeModel::ONNXRuntimeModel(file_name_t path_to_model, int intra_op_num_threads,
				   int num_classes, int inter_op_num_threads,
				   const std::string &amp;use_gpu_, int device_id, bool use_parallel,
				   float nms_th, float conf_th)
	: intra_op_num_threads_(intra_op_num_threads),
	  inter_op_num_threads_(inter_op_num_threads),
	  use_gpu(use_gpu_),
	  device_id_(device_id),
	  use_parallel_(use_parallel),
	  nms_thresh_(nms_th),
	  bbox_conf_thresh_(conf_th),
	  num_classes_(num_classes)
{
	try {
		Ort::SessionOptions session_options;

		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		if (this-&gt;use_parallel_) {
			session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
			session_options.SetInterOpNumThreads(this-&gt;inter_op_num_threads_);
		} else {
			session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
		}
		session_options.SetIntraOpNumThreads(this-&gt;intra_op_num_threads_);

#ifdef _WIN32
		if (this-&gt;use_gpu == "cuda") {
			OrtCUDAProviderOptions cuda_option;
			cuda_option.device_id = this-&gt;device_id_;
			session_options.AppendExecutionProvider_CUDA(cuda_option);
		}
		if (this-&gt;use_gpu == "dml") {
			auto &amp;api = Ort::GetApi();
			OrtDmlApi *dmlApi = nullptr;
			Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION,
								      (const void **)&amp;dmlApi));
			Ort::ThrowOnError(dmlApi-&gt;SessionOptionsAppendExecutionProvider_DML(
				session_options, 0));
		}
#endif

		this-&gt;session_ = Ort::Session(this-&gt;env_, path_to_model.c_str(), session_options);
	} catch (std::exception &amp;e) {
		obs_log(LOG_ERROR, "Cannot load model: %s", e.what());
		throw e;
	}

	Ort::AllocatorWithDefaultOptions ort_alloc;

	// number of inputs
	size_t num_input = this-&gt;session_.GetInputCount();

	for (size_t i = 0; i &lt; num_input; i++) {
		auto input_info = this-&gt;session_.GetInputTypeInfo(i);
		auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
		auto input_shape = input_shape_info.GetShape();
		auto input_tensor_type = input_shape_info.GetElementType();

		// 验证输入形状维度
		if (input_shape.size() &gt;= 4) {
			this-&gt;input_h_.push_back((int)(input_shape[2]));
			this-&gt;input_w_.push_back((int)(input_shape[3]));
		} else {
			obs_log(LOG_ERROR, "Invalid input shape dimensions: %zu, expected at least 4", 
				input_shape.size());
			throw std::runtime_error("Invalid input shape, expected NCHW format");
		}

		// Allocate input memory buffer
		this-&gt;input_name_.push_back(
			std::string(this-&gt;session_.GetInputNameAllocated(i, ort_alloc).get()));
		size_t input_byte_count = sizeof(float) * input_shape_info.GetElementCount();
		std::unique_ptr&lt;uint8_t[]&gt; input_buffer =
			std::make_unique&lt;uint8_t[]&gt;(input_byte_count);
		auto input_memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

		this-&gt;input_tensor_.push_back(Ort::Value::CreateTensor(
			input_memory_info, input_buffer.get(), input_byte_count, input_shape.data(),
			input_shape.size(), input_tensor_type));
		this-&gt;input_buffer_.push_back(std::move(input_buffer));

		obs_log(LOG_INFO, "Input name: %s", this-&gt;input_name_[i].c_str());
		obs_log(LOG_INFO, "Input shape: %d %d %d %d", input_shape[0],
			input_shape.size() &gt; 1 ? input_shape[1] : 0,
			input_shape.size() &gt; 2 ? input_shape[2] : 0,
			input_shape.size() &gt; 3 ? input_shape[3] : 0);
	}

	// number of outputs
	size_t num_output = this-&gt;session_.GetOutputCount();

	for (size_t i = 0; i &lt; num_output; i++) {
		auto output_info = this-&gt;session_.GetOutputTypeInfo(i);
		auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
		auto output_shape = output_shape_info.GetShape();
		auto output_tensor_type = output_shape_info.GetElementType();

		this-&gt;output_shapes_.push_back(output_shape);

		// Allocate output memory buffer
		size_t output_byte_count = sizeof(float) * output_shape_info.GetElementCount();
		std::unique_ptr&lt;uint8_t[]&gt; output_buffer =
			std::make_unique&lt;uint8_t[]&gt;(output_byte_count);
		auto output_memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

		this-&gt;output_tensor_.push_back(Ort::Value::CreateTensor(
			output_memory_info, output_buffer.get(), output_byte_count,
			output_shape.data(), output_shape.size(), output_tensor_type));
		this-&gt;output_buffer_.push_back(std::move(output_buffer));

		this-&gt;output_name_.push_back(
			std::string(this-&gt;session_.GetOutputNameAllocated(i, ort_alloc).get()));

		obs_log(LOG_INFO, "Output name: %s", this-&gt;output_name_[i].c_str());
		obs_log(LOG_INFO, "Output shape: %d %d %d %d", output_shape[0],
			output_shape.size() &gt; 1 ? output_shape[1] : 0,
			output_shape.size() &gt; 2 ? output_shape[2] : 0,
			output_shape.size() &gt; 3 ? output_shape[3] : 0);
	}
}

cv::Mat ONNXRuntimeModel::static_resize(const cv::Mat &amp;img, const int input_index)
{
	if (input_index &lt; 0 || (size_t)input_index &gt;= input_w_.size() || 
	    (size_t)input_index &gt;= input_h_.size()) {
		obs_log(LOG_ERROR, "Invalid input_index: %d, vector sizes: input_w_=%zu, input_h_=%zu", 
			input_index, input_w_.size(), input_h_.size());
		throw std::out_of_range("Invalid input_index");
	}

	if (img.cols == 0 || img.rows == 0) {
		obs_log(LOG_ERROR, "Image dimensions cannot be zero");
		throw std::invalid_argument("Image dimensions cannot be zero");
	}

	float r = std::fminf((float)input_w_[input_index] / (float)img.cols,
			     (float)input_h_[input_index] / (float)img.rows);
	int unpad_w = (int)(r * (float)img.cols);
	int unpad_h = (int)(r * (float)img.rows);
	cv::Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size());
	cv::Mat out(input_h_[input_index], input_w_[input_index], CV_8UC3,
		    cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
	return out;
}

// for NCHW
void ONNXRuntimeModel::blobFromImage(const cv::Mat &amp;img, float *blob_data)
{
	if (blob_data == nullptr) {
		obs_log(LOG_ERROR, "blob_data is null");
		throw std::invalid_argument("blob_data cannot be null");
	}
	if (img.empty()) {
		obs_log(LOG_ERROR, "Input image is empty");
		throw std::invalid_argument("Input image cannot be empty");
	}

	size_t channels = 3;
	size_t img_h = img.rows;
	size_t img_w = img.cols;
	for (size_t c = 0; c &lt; channels; ++c) {
		for (size_t h = 0; h &lt; img_h; ++h) {
			for (size_t w = 0; w &lt; img_w; ++w) {
				blob_data[(int)(c * img_w * img_h + h * img_w + w)] =
					(float)img.ptr&lt;cv::Vec3b&gt;((int)h)[(int)w][(int)c];
			}
		}
	}
}

float ONNXRuntimeModel::intersection_area(const Object &amp;a, const Object &amp;b)
{
	cv::Rect_&lt;float&gt; inter = a.rect &amp; b.rect;
	return inter.area();
}

void ONNXRuntimeModel::qsort_descent_inplace(std::vector&lt;Object&gt; &amp;faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i &lt;= j) {
		while (faceobjects[i].prob &gt; p)
			++i;

		while (faceobjects[j].prob &lt; p)
			--j;

		if (i &lt;= j) {
			std::swap(faceobjects[i], faceobjects[j]);

			++i;
			--j;
		}
	}
	if (left &lt; j)
		qsort_descent_inplace(faceobjects, left, j);
	if (i &lt; right)
		qsort_descent_inplace(faceobjects, i, right);
}

void ONNXRuntimeModel::qsort_descent_inplace(std::vector&lt;Object&gt; &amp;objects)
{
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, (int)(objects.size() - 1));
}

void ONNXRuntimeModel::nms_sorted_bboxes(const std::vector&lt;Object&gt; &amp;objects,
					 std::vector&lt;int&gt; &amp;picked, const float nms_threshold)
{
	picked.clear();

	const size_t n = objects.size();

	std::vector&lt;float&gt; areas(n);
	for (size_t i = 0; i &lt; n; ++i) {
		areas[i] = objects[i].rect.area();
	}

	for (size_t i = 0; i &lt; n; ++i) {
		const Object &amp;a = objects[i];
		const size_t picked_size = picked.size();

		int keep = 1;
		for (size_t j = 0; j &lt; picked_size; ++j) {
			const Object &amp;b = objects[picked[j]];

			float inter_area = this-&gt;intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			
			if (union_area &gt; 0 &amp;&amp; inter_area / union_area &gt; nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back((int)i);
	}
}

void ONNXRuntimeModel::inference(const cv::Mat &amp;frame, const int input_index)
{
	if (input_index &lt; 0 || (size_t)input_index &gt;= input_buffer_.size()) {
		obs_log(LOG_ERROR, "Invalid input_index in inference: %d", input_index);
		throw std::out_of_range("Invalid input_index");
	}
	if (frame.empty()) {
		obs_log(LOG_ERROR, "Input frame is empty in inference");
		throw std::invalid_argument("Input frame cannot be empty");
	}

	cv::Mat pr_img = this-&gt;static_resize(frame, input_index);

	float *blob_data = (float *)(this-&gt;input_buffer_[input_index].get());
	blobFromImage(pr_img, blob_data);

	std::vector&lt;const char *&gt; input_names;
	for (size_t i = 0; i &lt; this-&gt;input_name_.size(); i++) {
		input_names.push_back(this-&gt;input_name_[i].c_str());
	}

	std::vector&lt;const char *&gt; output_names;
	for (size_t i = 0; i &lt; this-&gt;output_name_.size(); i++) {
		output_names.push_back(this-&gt;output_name_[i].c_str());
	}

	Ort::RunOptions run_options;
	this-&gt;session_.Run(run_options, input_names.data(), this-&gt;input_tensor_.data(),
			   this-&gt;input_tensor_.size(), output_names.data(),
			   this-&gt;output_tensor_.data(), this-&gt;output_tensor_.size());
}
