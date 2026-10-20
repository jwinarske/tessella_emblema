// What two drivers for one Adreno 640 cost, measured side by side.
//
//   aarch64-linux-gnu-gcc -O2 -o drvprof drvprof.c -lvulkan     (cross; see the baseline)
//   VK_ICD_FILENAMES=.../freedreno_icd.json ./drvprof all
//   VK_ICD_FILENAMES=/etc/vulkan/icd.d/adreno-vk.json ./drvprof all
//
// A render pass rather than dynamic rendering, because the blob has neither the 1.3 feature
// nor VK_KHR_dynamic_rendering -- which is the whole reason tessella_emblema's own benches
// cannot open a device on it. One binary, one loader, one board: the driver is the variable.
//
// Every figure is a p50 over repeats, because this board's load average sits above ten.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define SIDE 512
#define CHECK(expr)                                                                   \
    do {                                                                              \
        VkResult r_ = (expr);                                                         \
        if (r_ != VK_SUCCESS) {                                                       \
            fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #expr, (int)r_);   \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

#define REPEATS 15

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static double p50(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return v[n / 2];
}

static uint32_t *read_spv(const char *path, size_t *bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc((size_t)n);
    if (fread(w, 1, (size_t)n, f) != (size_t)n) { perror("read"); exit(1); }
    fclose(f);
    *bytes = (size_t)n;
    return w;
}

static uint32_t pick_memory(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type with 0x%x\n", (unsigned)want);
    exit(1);
}

struct gpu {
    VkInstance instance;
    VkPhysicalDevice pd;
    VkDevice device;
    VkQueue queue;
    uint32_t family;
    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkRenderPass pass;
    VkImage target;
    VkImageView view;
    VkFramebuffer fb;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkDescriptorSetLayout wide_set_layout;
    VkPipelineLayout wide_layout;
    VkDescriptorSet set;
    VkQueryPool queries;
    double timestamp_period;   // ns per tick, 0 when timestamps are unusable
    char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
};

static void open_gpu(struct gpu *g) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                              .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 .pApplicationInfo = &app };
    CHECK(vkCreateInstance(&ici, NULL, &g->instance));

    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(g->instance, &n, NULL));
    if (!n) { fprintf(stderr, "no Vulkan device\n"); exit(1); }
    VkPhysicalDevice *pds = calloc(n, sizeof *pds);
    CHECK(vkEnumeratePhysicalDevices(g->instance, &n, pds));
    g->pd = pds[0];   // one device on the board; the desktop's first is its discrete part
    free(pds);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g->pd, &props);
    snprintf(g->name, sizeof g->name, "%s", props.deviceName);

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g->pd, &families, NULL);
    VkQueueFamilyProperties *fp = calloc(families, sizeof *fp);
    vkGetPhysicalDeviceQueueFamilyProperties(g->pd, &families, fp);
    g->family = UINT32_MAX;
    uint32_t valid_bits = 0;
    for (uint32_t i = 0; i < families; i++)
        if (fp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            g->family = i;
            valid_bits = fp[i].timestampValidBits;
            break;
        }
    free(fp);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo q = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = g->family, .queueCount = 1,
                                  .pQueuePriorities = &priority };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &q };
    CHECK(vkCreateDevice(g->pd, &dci, NULL, &g->device));
    vkGetDeviceQueue(g->device, g->family, 0, &g->queue);

    g->timestamp_period = (valid_bits == 0) ? 0.0 : (double)props.limits.timestampPeriod;

    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                     .queueFamilyIndex = g->family };
    CHECK(vkCreateCommandPool(g->device, &cpci, NULL, &g->pool));
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = g->pool,
                                         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    CHECK(vkAllocateCommandBuffers(g->device, &cbai, &g->cmd));

    // The target and its render pass.
    VkImageCreateInfo tci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D,
                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .extent = { SIDE, SIDE, 1 },
                              .mipLevels = 1, .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    CHECK(vkCreateImage(g->device, &tci, NULL, &g->target));
    VkMemoryRequirements need;
    vkGetImageMemoryRequirements(g->device, g->target, &need);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = need.size,
                                 .memoryTypeIndex = pick_memory(g->pd, need.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
    VkDeviceMemory tmem;
    CHECK(vkAllocateMemory(g->device, &mai, NULL, &tmem));
    CHECK(vkBindImageMemory(g->device, g->target, tmem, 0));
    VkImageViewCreateInfo ivci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                   .image = g->target, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    CHECK(vkCreateImageView(g->device, &ivci, NULL, &g->view));

    VkAttachmentDescription color = { .format = VK_FORMAT_R8G8B8A8_UNORM,
                                       .samples = VK_SAMPLE_COUNT_1_BIT,
                                       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                       .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                       .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                       .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
    VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = 1, .pColorAttachments = &ref };
    VkRenderPassCreateInfo rpci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                    .attachmentCount = 1, .pAttachments = &color,
                                    .subpassCount = 1, .pSubpasses = &sub };
    CHECK(vkCreateRenderPass(g->device, &rpci, NULL, &g->pass));
    VkFramebufferCreateInfo fbci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                     .renderPass = g->pass, .attachmentCount = 1,
                                     .pAttachments = &g->view,
                                     .width = SIDE, .height = SIDE, .layers = 1 };
    CHECK(vkCreateFramebuffer(g->device, &fbci, NULL, &g->fb));

    // One set layout for every pipeline here: a sampled image and a sampler, which only the
    // sampling shader reads. A layout wider than the module is valid.
    VkDescriptorSetLayoutBinding binds[2] = {
        { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = binds };
    CHECK(vkCreateDescriptorSetLayout(g->device, &dslci, NULL, &g->set_layout));
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                        .setLayoutCount = 1, .pSetLayouts = &g->set_layout };
    CHECK(vkCreatePipelineLayout(g->device, &plci, NULL, &g->layout));

    VkDescriptorPoolSize sizes[2] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1 },
                                      { VK_DESCRIPTOR_TYPE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = sizes };
    VkDescriptorPool dpool;
    CHECK(vkCreateDescriptorPool(g->device, &dpci, NULL, &dpool));
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                         .descriptorPool = dpool, .descriptorSetCount = 1,
                                         .pSetLayouts = &g->set_layout };
    CHECK(vkAllocateDescriptorSets(g->device, &dsai, &g->set));

    // And a layout wide enough for a real tessella_emblema family: its widest set is six
    // storage buffers with three sampled images and three samplers, and a layout wider than the
    // module is valid. Pipeline creation against a layout that does *not* cover the module's
    // resources is invalid usage, which some drivers answer with a segfault.
    VkDescriptorSetLayoutBinding wide[14];
    for (uint32_t i = 0; i < 6; i++)
        wide[i] = (VkDescriptorSetLayoutBinding){ i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                  VK_SHADER_STAGE_VERTEX_BIT |
                                                  VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    for (uint32_t i = 0; i < 4; i++)
        wide[6 + i] = (VkDescriptorSetLayoutBinding){ 6 + i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                                                      VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    for (uint32_t i = 0; i < 4; i++)
        wide[10 + i] = (VkDescriptorSetLayoutBinding){ 10 + i, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                                                       VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo wlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 14, .pBindings = wide };
    CHECK(vkCreateDescriptorSetLayout(g->device, &wlci, NULL, &g->wide_set_layout));
    VkPipelineLayoutCreateInfo wplci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                         .setLayoutCount = 1, .pSetLayouts = &g->wide_set_layout };
    CHECK(vkCreatePipelineLayout(g->device, &wplci, NULL, &g->wide_layout));

    if (g->timestamp_period > 0.0) {
        VkQueryPoolCreateInfo qpci = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                       .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 };
        if (vkCreateQueryPool(g->device, &qpci, NULL, &g->queries) != VK_SUCCESS)
            g->timestamp_period = 0.0;
    }
}

// A texture for the sampling shader, filled with something that is not one color.
static void stage_texture(struct gpu *g) {
    VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D,
                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .extent = { SIDE, SIDE, 1 },
                              .mipLevels = 1, .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_LINEAR,
                              .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
                              .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED };
    VkImage image;
    CHECK(vkCreateImage(g->device, &ici, NULL, &image));
    VkMemoryRequirements need;
    vkGetImageMemoryRequirements(g->device, image, &need);
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                 .allocationSize = need.size,
                                 .memoryTypeIndex = pick_memory(g->pd, need.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
    VkDeviceMemory mem;
    CHECK(vkAllocateMemory(g->device, &mai, NULL, &mem));
    CHECK(vkBindImageMemory(g->device, image, mem, 0));
    VkImageSubresource sr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout layout;
    vkGetImageSubresourceLayout(g->device, image, &sr, &layout);
    void *mapped = NULL;
    CHECK(vkMapMemory(g->device, mem, 0, need.size, 0, &mapped));
    for (uint32_t y = 0; y < SIDE; y++) {
        unsigned char *row = (unsigned char *)mapped + layout.offset + (VkDeviceSize)y * layout.rowPitch;
        for (uint32_t x = 0; x < SIDE; x++) {
            row[x * 4 + 0] = (unsigned char)(x ^ y);
            row[x * 4 + 1] = (unsigned char)(x + y);
            row[x * 4 + 2] = (unsigned char)(x * 3);
            row[x * 4 + 3] = 255;
        }
    }
    vkUnmapMemory(g->device, mem);

    VkImageViewCreateInfo ivci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                   .image = image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = VK_FORMAT_R8G8B8A8_UNORM,
                                   .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    VkImageView view;
    CHECK(vkCreateImageView(g->device, &ivci, NULL, &view));
    VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
                                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                .maxLod = 0.0f };
    VkSampler sampler;
    CHECK(vkCreateSampler(g->device, &sci, NULL, &sampler));

    // PREINITIALIZED to SHADER_READ_ONLY, which needs a submitted barrier.
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CHECK(vkBeginCommandBuffer(g->cmd, &begin));
    VkImageMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                     .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
                                     .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                     .image = image,
                                     .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
    vkCmdPipelineBarrier(g->cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &barrier);
    CHECK(vkEndCommandBuffer(g->cmd));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &g->cmd };
    CHECK(vkQueueSubmit(g->queue, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(g->queue));

    VkDescriptorImageInfo image_info = { .imageView = view,
                                         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo sampler_info = { .sampler = sampler };
    VkWriteDescriptorSet writes[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g->set, .dstBinding = 0,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .pImageInfo = &image_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g->set, .dstBinding = 1,
          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .pImageInfo = &sampler_info },
    };
    vkUpdateDescriptorSets(g->device, 2, writes, 0, NULL);
}

static VkShaderModule module_of(struct gpu *g, const char *path) {
    size_t bytes = 0;
    uint32_t *words = read_spv(path, &bytes);
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = bytes, .pCode = words };
    VkShaderModule m;
    CHECK(vkCreateShaderModule(g->device, &smci, NULL, &m));
    free(words);
    return m;
}

// One pipeline from two single-entry-point modules. `tweak` varies a harmless piece of state
// so a driver cannot hand back a pipeline it built earlier.
static VkPipeline pipeline_of_named(struct gpu *g, VkShaderModule vs, VkShaderModule fs, int tweak,
                                    VkPipelineCache cache, const char *vs_entry,
                                    const char *fs_entry, VkPipelineLayout layout) {
    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = vs_entry },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = fs_entry },
    };
    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkViewport viewport = { 0, 0, (float)SIDE, (float)SIDE, 0.0f, 1.0f };
    VkRect2D scissor = { { 0, 0 }, { SIDE, SIDE } };
    VkPipelineViewportStateCreateInfo vp = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor };
    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
        .depthBiasConstantFactor = (float)tweak };
    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineColorBlendAttachmentState blend = { .colorWriteMask = 0xf };
    VkPipelineColorBlendStateCreateInfo cb = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &blend };
    VkGraphicsPipelineCreateInfo gpci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                          .stageCount = 2, .pStages = stages,
                                          .pVertexInputState = &vi, .pInputAssemblyState = &ia,
                                          .pViewportState = &vp, .pRasterizationState = &rs,
                                          .pMultisampleState = &ms, .pColorBlendState = &cb,
                                          .layout = layout, .renderPass = g->pass };
    VkPipeline p;
    CHECK(vkCreateGraphicsPipelines(g->device, cache, 1, &gpci, NULL, &p));
    return p;
}

static VkPipeline pipeline_of(struct gpu *g, VkShaderModule vs, VkShaderModule fs, int tweak,
                              VkPipelineCache cache) {
    return pipeline_of_named(g, vs, fs, tweak, cache, "vs_main", "fs_main", g->layout);
}

// What one real family's module pair costs to compile, which the shaders above cannot say:
// they are 88 to 358 words and a family is one to three thousand.
static void bench_module(struct gpu *g, const char *vs_path, const char *fs_path) {
    VkShaderModule vs = module_of(g, vs_path);
    VkShaderModule fs = module_of(g, fs_path);

    // Twice over: with no cache object, and with one. Each iteration varies a piece of harmless
    // state, so these are *different pipelines sharing one module pair* -- which is exactly what
    // tessella_emblema builds, 168 pipelines out of 66 pairs. The second column is what a cache
    // object is worth to that.
    double bare[REPEATS], cached[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        double start = now_us();
        VkPipeline p = pipeline_of_named(g, vs, fs, r + 1, VK_NULL_HANDLE, "vertex_main",
                                         "fragment_main", g->wide_layout);
        bare[r] = now_us() - start;
        vkDestroyPipeline(g->device, p, NULL);
    }
    VkPipelineCacheCreateInfo pcci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    VkPipelineCache cache;
    CHECK(vkCreatePipelineCache(g->device, &pcci, NULL, &cache));
    for (int r = 0; r < REPEATS; r++) {
        double start = now_us();
        VkPipeline p = pipeline_of_named(g, vs, fs, r + 1, cache, "vertex_main", "fragment_main",
                                         g->wide_layout);
        cached[r] = now_us() - start;
        vkDestroyPipeline(g->device, p, NULL);
    }
    vkDestroyPipelineCache(g->device, cache, NULL);

    printf("  %-26s  no cache %9.1f first %9.1f p50   cached %9.1f first %8.1f p50\n", vs_path,
           bare[0], p50(bare + 1, REPEATS - 1), cached[0], p50(cached + 1, REPEATS - 1));
    vkDestroyShaderModule(g->device, fs, NULL);
    vkDestroyShaderModule(g->device, vs, NULL);
}

// One submitted pass of `draws` draws with `pipeline`, timed on the GPU where timestamps work
// and on the wall either way. Returns wall microseconds; *gpu_us is -1 without timestamps.
static double one_pass(struct gpu *g, VkPipeline pipeline, int draws, int vertices,
                       double *gpu_us) {
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CHECK(vkBeginCommandBuffer(g->cmd, &begin));
    if (g->timestamp_period > 0.0) {
        vkCmdResetQueryPool(g->cmd, g->queries, 0, 2);
        vkCmdWriteTimestamp(g->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g->queries, 0);
    }
    VkClearValue clear = { .color = { .float32 = { 0.2f, 0.4f, 0.6f, 1.0f } } };
    VkRenderPassBeginInfo rpbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                   .renderPass = g->pass, .framebuffer = g->fb,
                                   .renderArea = { { 0, 0 }, { SIDE, SIDE } },
                                   .clearValueCount = 1, .pClearValues = &clear };
    vkCmdBeginRenderPass(g->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g->layout, 0, 1, &g->set,
                            0, NULL);
    for (int d = 0; d < draws; d++) vkCmdDraw(g->cmd, (uint32_t)vertices, 1, 0, 0);
    vkCmdEndRenderPass(g->cmd);
    if (g->timestamp_period > 0.0)
        vkCmdWriteTimestamp(g->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g->queries, 1);
    CHECK(vkEndCommandBuffer(g->cmd));

    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1, .pCommandBuffers = &g->cmd };
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    CHECK(vkCreateFence(g->device, &fci, NULL, &fence));
    double start = now_us();
    CHECK(vkQueueSubmit(g->queue, 1, &si, fence));
    CHECK(vkWaitForFences(g->device, 1, &fence, VK_TRUE, UINT64_MAX));
    double wall = now_us() - start;
    vkDestroyFence(g->device, fence, NULL);

    *gpu_us = -1.0;
    if (g->timestamp_period > 0.0) {
        uint64_t ticks[2] = { 0, 0 };
        if (vkGetQueryPoolResults(g->device, g->queries, 0, 2, sizeof ticks, ticks,
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT |
                                  VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS &&
            ticks[1] > ticks[0]) {
            *gpu_us = (double)(ticks[1] - ticks[0]) * g->timestamp_period / 1000.0;
        }
    }
    return wall;
}

static void bench_pipelines(struct gpu *g) {
    const char *names[] = { "fs-const", "fs-alu", "fs-tex" };
    double module_us[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        size_t bytes = 0;
        uint32_t *words = read_spv("fs-alu.spv", &bytes);
        VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                          .codeSize = bytes, .pCode = words };
        VkShaderModule m;
        double start = now_us();
        CHECK(vkCreateShaderModule(g->device, &smci, NULL, &m));
        module_us[r] = now_us() - start;
        vkDestroyShaderModule(g->device, m, NULL);
        free(words);
    }
    printf("  vkCreateShaderModule (fs-alu)   %9.1f us cold   %8.1f us p50\n", module_us[0],
           p50(module_us + 1, REPEATS - 1));

    VkShaderModule vs = module_of(g, "vs-quad.spv");
    for (unsigned i = 0; i < sizeof names / sizeof *names; i++) {
        char path[64];
        snprintf(path, sizeof path, "%s.spv", names[i]);
        VkShaderModule fs = module_of(g, path);
        // Cold and warm are different questions: a driver that caches compiled shaders --
        // Mesa does, in process and on disk -- answers the second create from the cache, and
        // the median of a loop over one shader is a cache hit rather than a compile.
        double create[REPEATS], cold;
        for (int r = 0; r < REPEATS; r++) {
            double start = now_us();
            VkPipeline p = pipeline_of(g, vs, fs, r + 1, VK_NULL_HANDLE);
            create[r] = now_us() - start;
            vkDestroyPipeline(g->device, p, NULL);
        }
        cold = create[0];
        printf("  vkCreateGraphicsPipelines %-10s  %9.1f us cold   %8.1f us p50\n", names[i],
               cold, p50(create + 1, REPEATS - 1));
        vkDestroyShaderModule(g->device, fs, NULL);
    }
    vkDestroyShaderModule(g->device, vs, NULL);
}

// Does the driver honor a VkPipelineCache -- in process, and across processes?
//
// The question the table without one leaves open. `--cache <path>` loads that file into a
// cache object before the first create and writes the cache back at the end, so the second
// invocation of this program is the case an application actually ships: a cache persisted from
// a previous run. Three numbers a shader: the first create against a cache that is empty (or
// preloaded), the repeats, and what the cache weighs.
static void bench_cache(struct gpu *g, const char *cache_path) {
    unsigned char *initial = NULL;
    size_t initial_size = 0;
    if (cache_path) {
        FILE *f = fopen(cache_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long n = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (n > 0) {
                initial = malloc((size_t)n);
                initial_size = fread(initial, 1, (size_t)n, f);
            }
            fclose(f);
        }
    }
    printf("  pipeline cache: %s, %zu bytes loaded\n",
           cache_path ? cache_path : "(none, in process only)", initial_size);

    VkPipelineCacheCreateInfo pcci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                                       .initialDataSize = initial_size,
                                       .pInitialData = initial_size ? initial : NULL };
    VkPipelineCache cache;
    double start = now_us();
    VkResult made = vkCreatePipelineCache(g->device, &pcci, NULL, &cache);
    double create_us = now_us() - start;
    if (made != VK_SUCCESS) {
        // A driver may reject data it did not write; an empty cache still answers the
        // in-process half of the question.
        printf("  vkCreatePipelineCache rejected the data (%d), retrying empty\n", (int)made);
        pcci.initialDataSize = 0;
        pcci.pInitialData = NULL;
        CHECK(vkCreatePipelineCache(g->device, &pcci, NULL, &cache));
        initial_size = 0;
    }
    printf("  vkCreatePipelineCache     %9.1f us\n", create_us);

    const char *names[] = { "fs-const", "fs-alu", "fs-tex" };
    VkShaderModule vs = module_of(g, "vs-quad.spv");
    for (unsigned i = 0; i < sizeof names / sizeof *names; i++) {
        char path[64];
        snprintf(path, sizeof path, "%s.spv", names[i]);
        VkShaderModule fs = module_of(g, path);
        double create[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            double t = now_us();
            VkPipeline p = pipeline_of(g, vs, fs, r + 1, cache);
            create[r] = now_us() - t;
            vkDestroyPipeline(g->device, p, NULL);
        }
        printf("  with a cache  %-10s      %9.1f us first   %8.1f us p50\n", names[i],
               create[0], p50(create + 1, REPEATS - 1));
        vkDestroyShaderModule(g->device, fs, NULL);
    }
    vkDestroyShaderModule(g->device, vs, NULL);

    size_t written = 0;
    if (vkGetPipelineCacheData(g->device, cache, &written, NULL) == VK_SUCCESS) {
        printf("  vkGetPipelineCacheData    %9zu bytes\n", written);
        if (cache_path && written) {
            unsigned char *blob = malloc(written);
            if (vkGetPipelineCacheData(g->device, cache, &written, blob) == VK_SUCCESS) {
                FILE *f = fopen(cache_path, "wb");
                if (f) {
                    fwrite(blob, 1, written, f);
                    fclose(f);
                    printf("  wrote %zu bytes to %s\n", written, cache_path);
                }
            }
            free(blob);
        }
    } else {
        printf("  vkGetPipelineCacheData    refused\n");
    }
    vkDestroyPipelineCache(g->device, cache, NULL);
    free(initial);
}

static void bench_record(struct gpu *g) {
    VkShaderModule vs = module_of(g, "vs-quad.spv");
    VkShaderModule fs = module_of(g, "fs-const.spv");
    VkPipeline p = pipeline_of(g, vs, fs, 1, VK_NULL_HANDLE);
    const int draws = 20000;
    double per_draw[REPEATS], per_bind[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        CHECK(vkBeginCommandBuffer(g->cmd, &begin));
        VkClearValue clear = { .color = { .float32 = { 0.2f, 0.4f, 0.6f, 1.0f } } };
        VkRenderPassBeginInfo rpbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                       .renderPass = g->pass, .framebuffer = g->fb,
                                       .renderArea = { { 0, 0 }, { SIDE, SIDE } },
                                       .clearValueCount = 1, .pClearValues = &clear };
        vkCmdBeginRenderPass(g->cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(g->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        double start = now_us();
        for (int d = 0; d < draws; d++) vkCmdDraw(g->cmd, 3, 1, 0, 0);
        per_draw[r] = (now_us() - start) / draws;
        start = now_us();
        for (int d = 0; d < draws; d++)
            vkCmdBindDescriptorSets(g->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g->layout, 0, 1,
                                    &g->set, 0, NULL);
        per_bind[r] = (now_us() - start) / draws;
        vkCmdEndRenderPass(g->cmd);
        CHECK(vkEndCommandBuffer(g->cmd));
        CHECK(vkResetCommandBuffer(g->cmd, 0));
    }
    printf("  vkCmdDraw                 %20.4f us a draw\n", p50(per_draw, REPEATS));
    printf("  vkCmdBindDescriptorSets   %20.4f us a bind\n", p50(per_bind, REPEATS));
    vkDestroyPipeline(g->device, p, NULL);
    vkDestroyShaderModule(g->device, fs, NULL);
    vkDestroyShaderModule(g->device, vs, NULL);
}

static void bench_submit(struct gpu *g) {
    VkShaderModule vs = module_of(g, "vs-quad.spv");
    VkShaderModule fs = module_of(g, "fs-const.spv");
    VkPipeline p = pipeline_of(g, vs, fs, 1, VK_NULL_HANDLE);
    double wall[REPEATS], gpu[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        double g_us = 0.0;
        wall[r] = one_pass(g, p, 1, 3, &g_us);
        gpu[r] = g_us;
    }
    printf("  one pass, one draw        %8.1f us wall p50", p50(wall, REPEATS));
    if (gpu[0] >= 0.0) printf("   %8.1f us gpu p50", p50(gpu, REPEATS));
    printf("\n");
    vkDestroyPipeline(g->device, p, NULL);
    vkDestroyShaderModule(g->device, fs, NULL);
    vkDestroyShaderModule(g->device, vs, NULL);
}

static void bench_gpu(struct gpu *g, const char *label, const char *vs_path, const char *fs_path,
                      int draws, int vertices) {
    VkShaderModule vs = module_of(g, vs_path);
    VkShaderModule fs = module_of(g, fs_path);
    VkPipeline p = pipeline_of(g, vs, fs, 1, VK_NULL_HANDLE);
    double wall[REPEATS], gpu[REPEATS];
    for (int r = 0; r < REPEATS; r++) {
        double g_us = 0.0;
        wall[r] = one_pass(g, p, draws, vertices, &g_us);
        gpu[r] = g_us;
    }
    double gpu_min = gpu[0], gpu_p50;
    for (int r = 1; r < REPEATS; r++) if (gpu[r] < gpu_min) gpu_min = gpu[r];
    gpu_p50 = p50(gpu, REPEATS);
    printf("  %-24s  %10.1f us wall p50", label, p50(wall, REPEATS));
    if (gpu[0] >= 0.0) printf("   gpu %10.1f min %10.1f p50", gpu_min, gpu_p50);
    printf("\n");
    vkDestroyPipeline(g->device, p, NULL);
    vkDestroyShaderModule(g->device, fs, NULL);
    vkDestroyShaderModule(g->device, vs, NULL);
}

int main(int argc, char **argv) {
    const char *what = argc > 1 ? argv[1] : "all";
    const char *cache_path = argc > 2 ? argv[2] : NULL;
    struct gpu g = { 0 };
    double start = now_us();
    open_gpu(&g);
    double opened = now_us() - start;
    printf("device: %s\n", g.name);
    printf("  instance + device + pass  %8.1f us (once)\n", opened);
    printf("  timestamps: %s\n", g.timestamp_period > 0.0 ? "yes" : "no");
    stage_texture(&g);

    int all = strcmp(what, "all") == 0;
    if (all || !strcmp(what, "pipeline")) bench_pipelines(&g);
    if (all || !strcmp(what, "cache")) bench_cache(&g, cache_path);
    if (!strcmp(what, "module")) {
        // Pairs of per-stage modules from this crate's own families, emitted by naga one entry
        // point at a time -- which is what the Adreno blob requires and what `pipelines::Stages`
        // does. Named on the command line, vertex then fragment.
        for (int i = 2; i + 1 < argc; i += 2) bench_module(&g, argv[i], argv[i + 1]);
    }
    if (all || !strcmp(what, "record")) bench_record(&g);
    if (all || !strcmp(what, "submit")) bench_submit(&g);
    if (all || !strcmp(what, "fill")) {
        bench_gpu(&g, "fill: 1 draw, const", "vs-quad.spv", "fs-const.spv", 1, 3);
        bench_gpu(&g, "fill: 8 draws, const", "vs-quad.spv", "fs-const.spv", 8, 3);
        bench_gpu(&g, "alu: 1 draw, 256 mad", "vs-quad.spv", "fs-alu.spv", 1, 3);
        bench_gpu(&g, "alu: 8 draws, 256 mad", "vs-quad.spv", "fs-alu.spv", 8, 3);
        // The same arithmetic unrolled. ir3 leaves the 256-iteration loop rolled and half its
        // instruction slots are nops, so this pair says what the loop costs rather than the
        // arithmetic.
        bench_gpu(&g, "alu: 1 draw, flat", "vs-quad.spv", "fs-alu-flat.spv", 1, 3);
        bench_gpu(&g, "alu: 8 draws, flat", "vs-quad.spv", "fs-alu-flat.spv", 8, 3);
        // And the same pair with a chain neither compiler can fold -- squaring is not affine.
        bench_gpu(&g, "sq: 1 draw, rolled", "vs-quad.spv", "fs-sq.spv", 1, 3);
        bench_gpu(&g, "sq: 1 draw, flat", "vs-quad.spv", "fs-sq-flat.spv", 1, 3);
    }
    if (all || !strcmp(what, "sample")) {
        bench_gpu(&g, "sample: 1 draw, 16 taps", "vs-quad.spv", "fs-tex.spv", 1, 3);
        bench_gpu(&g, "sample: 8 draws, 16 taps", "vs-quad.spv", "fs-tex.spv", 8, 3);
    }
    if (all || !strcmp(what, "vertex")) {
        bench_gpu(&g, "vertex: 1M degenerate", "vs-degenerate.spv", "fs-const.spv", 1, 1000002);
        bench_gpu(&g, "vertex: 4M degenerate", "vs-degenerate.spv", "fs-const.spv", 4, 1000002);
    }
    return 0;
}
