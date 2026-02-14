#include "obs-utils.h"
#include "plugin-support.h"

#include &lt;obs-module.h&gt;

/**
  * @brief Get RGBA from the stage surface
  *
  * @param tf  The filter data
  * @param width  The width of the stage surface (output)
  * @param height  The height of the stage surface (output)
  * @return true  if successful
  * @return false if unsuccessful
*/
bool getRGBAFromStageSurface(filter_data *tf, uint32_t &amp;width, uint32_t &amp;height)
{

	if (!obs_source_enabled(tf-&gt;source)) {
		return false;
	}

	obs_source_t *target = obs_filter_get_target(tf-&gt;source);
	if (!target) {
		return false;
	}
	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		return false;
	}
	gs_texrender_reset(tf-&gt;texrender);
	if (!gs_texrender_begin(tf-&gt;texrender, width, height)) {
		return false;
	}
	struct vec4 background;
	vec4_zero(&amp;background);
	gs_clear(GS_CLEAR_COLOR, &amp;background, 0.0f, 0);
	gs_ortho(0.0f, static_cast&lt;float&gt;(width), 0.0f, static_cast&lt;float&gt;(height), -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tf-&gt;texrender);

	if (tf-&gt;stagesurface) {
		uint32_t stagesurf_width = gs_stagesurface_get_width(tf-&gt;stagesurface);
		uint32_t stagesurf_height = gs_stagesurface_get_height(tf-&gt;stagesurface);
		if (stagesurf_width != width || stagesurf_height != height) {
			gs_stagesurface_destroy(tf-&gt;stagesurface);
			tf-&gt;stagesurface = nullptr;
		}
	}
	if (!tf-&gt;stagesurface) {
		tf-&gt;stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
	}
	gs_stage_texture(tf-&gt;stagesurface, gs_texrender_get_texture(tf-&gt;texrender));
	uint8_t *video_data;
	uint32_t linesize;
	if (!gs_stagesurface_map(tf-&gt;stagesurface, &amp;video_data, &amp;linesize)) {
		return false;
	}
	{
		std::lock_guard&lt;std::mutex&gt; lock(tf-&gt;inputBGRALock);
		tf-&gt;inputBGRA = cv::Mat(height, width, CV_8UC4, video_data, linesize);
	}
	gs_stagesurface_unmap(tf-&gt;stagesurface);
	return true;
}
