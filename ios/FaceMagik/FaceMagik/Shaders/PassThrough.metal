//
//  PassThrough.metal
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/23/20.
//  Taken and modified from PassThrough.metal from Project TrueDepthBackdrop and is provided
//  under MIT License.
//  Link to project: https://developer.apple.com/documentation/avfoundation/cameras_and_media_capture/enhancing_live_video_by_leveraging_truedepth_camera_data
//
//  Abstract:
//  Implements a passthrough shader used for previewing content.

#include <metal_stdlib>
using namespace metal;

// Vertex input/output structure for passing results from vertex shader to fragment shader
struct VertexIO
{
    float4 position [[position]];
    float2 textureCoord [[user(texturecoord)]];
};

// Vertex shader for a textured quad
vertex VertexIO vertexPassThrough(const device packed_float4 *pPosition  [[ buffer(0) ]],
                                  const device packed_float2 *pTexCoords [[ buffer(1) ]],
                                        uint                  vid        [[ vertex_id ]])
{
    VertexIO outVertex;
    
    outVertex.position = pPosition[vid];
    outVertex.textureCoord = pTexCoords[vid];
    
    return outVertex;
}

// Fragment shader for a textured quad
fragment half4 fragmentPassThrough(VertexIO        inputFragment [[ stage_in ]],
                                   texture2d<half> inputTexture  [[ texture(0) ]],
                                   sampler         samplr        [[ sampler(0) ]])
{
    return inputTexture.sample(samplr, inputFragment.textureCoord);
}



