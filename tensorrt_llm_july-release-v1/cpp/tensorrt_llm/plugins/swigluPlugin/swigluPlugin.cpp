/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/plugins/swigluPlugin/swigluPlugin.h"
#include "tensorrt_llm/kernels/swigluKernels.h"
using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::SwigluPluginCreator;
using nvinfer1::plugin::SwigluPlugin;

static const char* SWIGLU_PLUGIN_VERSION{"1"};
static const char* SWIGLU_PLUGIN_NAME{"Swiglu"};
PluginFieldCollection SwigluPluginCreator::mFC{};
std::vector<PluginField> SwigluPluginCreator::mPluginAttributes;

SwigluPlugin::SwigluPlugin(nvinfer1::DataType type)
    : mType(type)
{
}

// Parameterized constructor
SwigluPlugin::SwigluPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    PLUGIN_ASSERT(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SwigluPlugin::clone() const noexcept
{
    auto* plugin = new SwigluPlugin(mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs SwigluPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    DimsExprs ret{};
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs > 0);
        DimsExprs output = inputs[0];
        PLUGIN_VALIDATE(output.nbDims >= 3);
        output.d[2] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[0].d[2], *exprBuilder.constant(2));
        ret = output;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
    //return inputs[outputIndex];
}

bool SwigluPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(0 <= pos && pos < 5);
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void SwigluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t SwigluPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}


int SwigluPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     mat1 [M, N*2] 
    // outputs
    //     mat [M, N]

    int32_t const gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    int32_t const nHalfHiddenSize = inputDesc[0].dims.d[2] / 2; // HHS

    if (mType == DataType::kHALF)
    {

        const half* input = reinterpret_cast<const half*>(inputs[0]);
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeSwiglu(output, input, gridSize, nHalfHiddenSize, stream);
    }
    else if (mType == DataType::kFLOAT)
    {

        const float* input = reinterpret_cast<const float*>(inputs[0]);
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeSwiglu(output, input, gridSize, nHalfHiddenSize, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        const __nv_bfloat16* input = reinterpret_cast<const __nv_bfloat16*>(inputs[0]);
        __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
        invokeSwiglu(output, input, gridSize, nHalfHiddenSize, stream);
    }
#endif

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType SwigluPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* SwigluPlugin::getPluginType() const noexcept
{
    return SWIGLU_PLUGIN_NAME;
}

const char* SwigluPlugin::getPluginVersion() const noexcept
{
    return SWIGLU_PLUGIN_VERSION;
}

int SwigluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int SwigluPlugin::initialize() noexcept
{
    return 0;
}

void SwigluPlugin::destroy() noexcept
{
    delete this;
}

size_t SwigluPlugin::getSerializationSize() const noexcept
{
    return sizeof(mType);
}

void SwigluPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void SwigluPlugin::terminate() noexcept {}

void SwigluPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SwigluPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

SwigluPluginCreator::SwigluPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SwigluPluginCreator::getPluginName() const noexcept
{
    return SWIGLU_PLUGIN_NAME;
}

const char* SwigluPluginCreator::getPluginVersion() const noexcept
{
    return SWIGLU_PLUGIN_VERSION;
}

const PluginFieldCollection* SwigluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SwigluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int transA, transB;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        
        if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new SwigluPlugin(type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SwigluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SwigluPlugin::destroy()
    try
    {
        auto* obj = new SwigluPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void SwigluPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SwigluPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
