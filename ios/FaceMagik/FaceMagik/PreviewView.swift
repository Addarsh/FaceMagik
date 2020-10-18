//
//  PreviewView.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 10/18/20.
//

import Foundation
import UIKit
import AVFoundation

class PreviewView: UIView {
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
    
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }
}
