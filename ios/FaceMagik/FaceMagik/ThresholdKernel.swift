//
//  ThresholdKernel.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/15/20.
//

import Foundation
import CoreImage
import MetalPerformanceShaders


class ThresholdImageProcessorKernel: CIImageProcessorKernel {
    override class func process(with inputs: [CIImageProcessorInput]?, arguments: [String : Any]?, output: CIImageProcessorOutput) throws {
    guard
        let commandBuffer = output.metalCommandBuffer,
        let input = inputs?.first,
        let sourceTexture = input.metalTexture,
        let destinationTexture = output.metalTexture,
        let thresholdValue = arguments?["thresholdValue"] as? Float,
        let device = arguments?["device"] as? MTLDevice else  {
            return
        }
    let threshold = MPSImageThresholdBinary(
        device: device,
        thresholdValue: thresholdValue,
        maximumValue: 1.0,
        linearGrayColorTransform: nil)
    threshold.encode(
        commandBuffer: commandBuffer,
        sourceTexture: sourceTexture,
        destinationTexture: destinationTexture)
    }
}
