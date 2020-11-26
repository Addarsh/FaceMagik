//
//  ViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 10/16/20.
//

import UIKit
import Photos
import CoreImage.CIFilterBuiltins

class ViewController: UIViewController {
    // Picker variables.
    let picker = UIImagePickerController()
    @IBOutlet var overlayView: UIView!
    var capturedImage: UIImage!
    let imageViewSegue = "imageView"
    
    // AVCaptureSession variables.
    @objc var cameraDevice: AVCaptureDevice!
    var sessionQueue: DispatchQueue!
    var captureSession =  AVCaptureSession()
    @IBOutlet weak private var previewView: PreviewMetalView!
    let photoProcessor = PhotoProcessor()
    let videoOutput = AVCaptureVideoDataOutput()
    let depthDataOutput = AVCaptureDepthDataOutput()
    private var outputSynchronizer: AVCaptureDataOutputSynchronizer!
    var depthCutOff: Float = 1.0
    let dataOutputQueue = DispatchQueue(label: "com.Addarsh.FaceMagik")
    let FRAME_RATE = 20

    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.previewView.rotation = .rotate180Degrees
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        self.sessionQueue = DispatchQueue(label: "session queue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem, target: .none)
        
        // Setup face detection request.
        self.photoProcessor.prepareDetectionRequest()
        
        // Setup video capture session.
        self.setupVideoCaptureSession()
        
        self.sessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidLoad()
        if !self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.startRunning()
            }
        }
    }
    
    @objc func appMovedToBackground() {
        if self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.stopRunning()
            }
        }
    }
    
    @objc func appMovedToForeground() {
        if !self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.startRunning()
            }
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if self.captureSession.isRunning {
            self.captureSession.stopRunning()
        }
    }

    // startPicker starts a UIImagePickerController session.
    @IBAction func startPicker() {
        self.picker.sourceType = .camera
        self.picker.cameraDevice = .front
        self.picker.showsCameraControls = false
        self.picker.cameraFlashMode = .off
        self.picker.delegate = self
        self.picker.cameraOverlayView = overlayView
        
        self.picker.cameraViewTransform = CGAffineTransform(translationX: 0, y: 120)
        
        present(self.picker, animated: true)
    }
    
    // dismissPicker dismisses given picker and -re-starts capture session.
    @IBAction func dismissPicker() {
        DispatchQueue.main.async {
            self.captureSession.startRunning()
        }
        self.picker.dismiss(animated: true)
    }
    
    // Picker take Picture.
    @IBAction func takePicture() {
        self.picker.takePicture()
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == self.imageViewSegue {
            guard let destVC = segue.destination as? ImageViewController else {
                return
            }
            destVC.image = capturedImage
        }
    }
    
    // setupVideoCaptureSession sets up a capture session to capture video.
    func setupVideoCaptureSession() {
        self.captureSession.beginConfiguration()
        
        // Add capture session input.
        guard let dev = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            return
        }
        self.cameraDevice = dev
        
        // Add capture session input.
        guard let captureInput = try? AVCaptureDeviceInput(device: self.cameraDevice), self.captureSession.canAddInput(captureInput) else {
            return
        }
        self.captureSession.addInput(captureInput)
        
        // Add capture session output.
        self.videoOutput.alwaysDiscardsLateVideoFrames = true
        guard self.captureSession.canAddOutput(self.videoOutput) else {
            return
        }
        
        // Set sRGB as default color space.
        self.captureSession.automaticallyConfiguresCaptureDeviceForWideColor = false
        self.captureSession.sessionPreset = .hd1280x720
        self.captureSession.addOutput(self.videoOutput)
            
        if let videoConnection = self.videoOutput.connection(with: .video) {
            videoConnection.videoOrientation = .portrait
            videoConnection.isEnabled = true
        }
        
        // Set sRGB as default color space.
        do {
            try self.cameraDevice.lockForConfiguration()
            self.cameraDevice.activeColorSpace = .sRGB
            self.cameraDevice.unlockForConfiguration()
        } catch {
            print("Error! Could not lock device for configuration: \(error)")
            return
        }
        
        // Add depth data output.
        //self.depthDataOutput.alwaysDiscardsLateDepthData = true
        self.depthDataOutput.isFilteringEnabled = true
        self.captureSession.addOutput(self.depthDataOutput)
        
        if let depthConnection = self.depthDataOutput.connection(with: .depthData) {
            depthConnection.videoOrientation = .portrait
            depthConnection.isEnabled = true
        }
        
        // Search for highest resolution with floating-point depth values
        let depthFormats = self.cameraDevice.activeFormat.supportedDepthDataFormats
        let depth32formats = depthFormats.filter({
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32
        })
        if depth32formats.isEmpty {
            print("Error! Device does not support Float32 depth format")
            return
        }
        
        let selectedFormat = depth32formats.max(by: { first, second in
            CMVideoFormatDescriptionGetDimensions(first.formatDescription).width <
                CMVideoFormatDescriptionGetDimensions(second.formatDescription).width })
        
        do {
            try self.cameraDevice.lockForConfiguration()
            self.cameraDevice.activeDepthDataFormat = selectedFormat
            self.cameraDevice.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: Int32(FRAME_RATE))
            self.cameraDevice.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: Int32(FRAME_RATE))
            self.cameraDevice.unlockForConfiguration()
        } catch {
            print("Error! Could not lock device for configuration: \(error)")
            return
        }
        
        // Use an AVCaptureDataOutputSynchronizer to synchronize the video data and depth data outputs.
        // The first output in the dataOutputs array, in this case the AVCaptureVideoDataOutput, is the "master" output.
        self.outputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [self.videoOutput, self.depthDataOutput])
        self.outputSynchronizer.setDelegate(self, queue: self.dataOutputQueue)
        
        self.captureSession.commitConfiguration()
    }
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        guard let uiimage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else {
            return
        }
        self.capturedImage = uiimage
        
        performSegue(withIdentifier: self.imageViewSegue, sender: nil)
        picker.dismiss(animated: true, completion: nil)
    }
}

extension ViewController: AVCaptureDataOutputSynchronizerDelegate {
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        
        // Check video frame.
        guard let syncedVideoData = synchronizedDataCollection.synchronizedData(for: self.videoOutput) as? AVCaptureSynchronizedSampleBufferData else {
            return
        }
        // Check depth data frame.
        guard let syncedDepthData = synchronizedDataCollection.synchronizedData(for: self.depthDataOutput) as? AVCaptureSynchronizedDepthData else {
            return
        }
        if syncedVideoData.sampleBufferWasDropped || syncedDepthData.depthDataWasDropped {
            return
        }

        guard let videoPixelBuffer = CMSampleBufferGetImageBuffer(syncedVideoData.sampleBuffer) else {
            print ("Could not convert video sample buffer to cvpixelbuffer")
            return
        }
        let mainCIImage = CIImage(cvPixelBuffer: videoPixelBuffer)
        let depthPixelBuffer = syncedDepthData.depthData.depthDataMap
        
        // detect face.
        self.photoProcessor.detectFace(mainCIImage)
        self.photoProcessor.semaphore.wait()
        if photoProcessor.numFaces == 0 {
            return
        }
        
        // compute depth mask.
        self.depthCutOff = computeDepthCutoff(depthPixelBuffer, mainCIImage.extent.width)
        guard var depthMaskCIImage = convertDepthMapToMask(CIImage(cvPixelBuffer: depthPixelBuffer)) else {
            print ("Could not convert depth mask to CIImage")
            return
        }
        
        let scale = CGAffineTransform(scaleX:  mainCIImage.extent.width/CGFloat(CVPixelBufferGetWidth(depthPixelBuffer)), y: mainCIImage.extent.height/CGFloat(CVPixelBufferGetHeight(depthPixelBuffer)))
        depthMaskCIImage = depthMaskCIImage.transformed(by: scale)
    
        
        self.photoProcessor.computeFinalFaceMask(depthMaskCIImage)
        self.photoProcessor.calculateOverExposedPointsGPU()
        
        self.previewView.image = self.photoProcessor.overExposedImage()
    }
    
    // computeDepthCutoff returns the depth value at the center of the detected face.
    func computeDepthCutoff(_ depthPixelBuffer: CVPixelBuffer, _ mainWidth: CGFloat) -> Float {
        guard let face = self.photoProcessor.faceBoundsRect else {
            return -1.0
        }
        let center = CGPoint(x: face.midX, y: face.minY)
        let scale = CGFloat(CVPixelBufferGetWidth(depthPixelBuffer))/mainWidth
        let pixelX = Int((center.x * scale).rounded())
        let pixelY = Int((center.y * scale).rounded())
        
        CVPixelBufferLockBaseAddress(depthPixelBuffer, .readOnly)
        
        let rowData = CVPixelBufferGetBaseAddress(depthPixelBuffer)! + pixelY * CVPixelBufferGetBytesPerRow(depthPixelBuffer)
        let faceCenterDepth = rowData.assumingMemoryBound(to: Float32.self)[pixelX]
        CVPixelBufferUnlockBaseAddress(depthPixelBuffer, .readOnly)
        
        return faceCenterDepth
    }
    
    // convertDepthMapToMask returns a depth map mask using a depth cutoff (already computed).
    // Every pixel below cutoff is converted to 1. otherwise it's 0.
    func convertDepthMapToMask(_ depthMaskCIImage:CIImage) -> CIImage? {
        let s :CGFloat = -10
        let b = -s*CGFloat(self.depthCutOff+0.25)
        
        guard let mat = CIFilter(name: "CIColorMatrix", parameters: ["inputImage": depthMaskCIImage, "inputRVector": CIVector(x: s, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: s, z: 0, w: 0),"inputBVector": CIVector(x: 0, y: 0, z: s, w: 0),"inputBiasVector": CIVector(x: b, y: b, z: b, w: 0)]) else {
            print ("Could not construct CIFilter")
            return nil
        }
        let clamp = CIFilter.colorClamp()
        clamp.inputImage = mat.outputImage
        return clamp.outputImage
    }
}
