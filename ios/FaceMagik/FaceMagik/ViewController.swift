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
    @IBOutlet var maskSwitch: UISwitch!
    
    // AVCaptureSession variables.
    @objc var cameraDevice: AVCaptureDevice!
    var sessionQueue: DispatchQueue!
    var captureSession =  AVCaptureSession()
    var captureOutput: AVCapturePhotoOutput!
    @IBOutlet private var previewLayer: PreviewView!
    var exposureObservation: NSKeyValueObservation?
    var tempObservation: NSKeyValueObservation?
    var isoObservation: NSKeyValueObservation?
    @IBOutlet var exposureLabel: UILabel!
    @IBOutlet var tempLabel: UILabel!
    @IBOutlet var isoLabel: UILabel!
    var currentCamera: AVCaptureDevice.Position = .unspecified
    let photoProcessor = PhotoProcessor()
    let videoOutput = AVCaptureVideoDataOutput()
    let depthDataOutput = AVCaptureDepthDataOutput()
    private var outputSynchronizer: AVCaptureDataOutputSynchronizer!
    var depthCutOff: Float = 1.0
    let dataOutputQueue = DispatchQueue(label: "com.Addarsh.FaceMagik")
    let FRAME_RATE = 10
    
    // Layer UI for drawing Vision results
    var videoResolution: CGSize!
    var detectionOverlayLayer: CALayer?
    var detectedFaceLayer: CAShapeLayer?
    var detecteddLandMarksLayer: CAShapeLayer?

    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.maskSwitch.isOn = false
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        self.sessionQueue = DispatchQueue(label: "session queue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem, target: .none)
        
        self.setupPhotoCaptureSession()
        
        // Setup video capture session.
        //self.setupVideoCaptureSession()
        //self.setupVisionDrawingLayers()
        
        // Setup face detection request.
        self.photoProcessor.prepareDetectionRequest()
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
    
    // Start recording video.
    @IBAction func startRecording() {
        let photoSettings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
        photoSettings.flashMode = .off
        if self.currentCamera == .front {
            photoSettings.isDepthDataDeliveryEnabled = true
            photoSettings.isPortraitEffectsMatteDeliveryEnabled = true
        }
        
        self.captureOutput.capturePhoto(with: photoSettings, delegate: self)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == self.imageViewSegue {
            guard let destVC = segue.destination as? ImageViewController else {
                return
            }
            destVC.image = capturedImage
            destVC.overExposedPercent = self.photoProcessor.overExposedPercent()
        }
    }
    
    @IBAction func switchSessions() {
        self.stopObservingDevice()
        
        guard let currentCameraInput = self.captureSession.inputs.first else {
            return
        }
        self.captureSession.beginConfiguration()
        
        self.captureSession.removeInput(currentCameraInput)
        
        var newDev: AVCaptureDevice?
        var newPosition: AVCaptureDevice.Position = .unspecified
        if let input = currentCameraInput as? AVCaptureDeviceInput {
            if input.device.position == .front {
                newDev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                newPosition = .back
            } else {
                newDev = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front)
                newPosition = .front
            }
        }
        guard let dev = newDev else {
            return
        }
        self.cameraDevice = dev
        
        guard let captureInput = try? AVCaptureDeviceInput(device: self.cameraDevice), self.captureSession.canAddInput(captureInput) else {
            return
        }
        self.captureSession.addInput(captureInput)
        
        if newPosition == .front {
            self.captureOutput.isDepthDataDeliveryEnabled = true
            self.captureOutput.isPortraitEffectsMatteDeliveryEnabled = true
        } else {
            self.captureOutput.isDepthDataDeliveryEnabled = false
            self.captureOutput.isPortraitEffectsMatteDeliveryEnabled = false
        }
        
        if let photoConnection = self.captureOutput.connection(with: .video) {
            photoConnection.videoOrientation = .portrait
        }
        
        self.captureSession.commitConfiguration()
        self.currentCamera = newPosition

        self.observeDevice()
    }
    
    // setupPhotoCaptureSession sets up a capture session to capture photos.
    func setupPhotoCaptureSession() {
        self.captureSession.beginConfiguration()
        self.previewLayer.videoPreviewLayer.session = self.captureSession
        
        // Add capture session input.
        guard let dev = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            return
        }
        self.cameraDevice = dev
        self.currentCamera = .front
        
        // Add capture session input.
        guard let captureInput = try? AVCaptureDeviceInput(device: self.cameraDevice), self.captureSession.canAddInput(captureInput) else {
            return
        }
        self.captureSession.addInput(captureInput)
        
        // Add capture session output.
        let photoOutput = AVCapturePhotoOutput()
        guard self.captureSession.canAddOutput(photoOutput) else {
            return
        }
        
        // Set sRGB as default color space.
        self.captureSession.automaticallyConfiguresCaptureDeviceForWideColor = false
        self.captureSession.sessionPreset = .hd1280x720
        self.captureSession.addOutput(photoOutput)
            
        if let photoConnection = photoOutput.connection(with: .video) {
            photoConnection.videoOrientation = .portrait
        }
        
        photoOutput.isDepthDataDeliveryEnabled = true
        photoOutput.isPortraitEffectsMatteDeliveryEnabled = true
        self.captureOutput = photoOutput
        
        // Set sRGB as default color space.
        do {
            try self.cameraDevice.lockForConfiguration()
            self.cameraDevice.activeColorSpace = .sRGB
            self.cameraDevice.unlockForConfiguration()
        } catch {
            print("Error! Could not lock device for configuration: \(error)")
            return
        }
        
        self.captureSession.commitConfiguration()
        
        self.observeDevice()
        
        self.sessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    // setupVideoCaptureSession sets up a capture session to capture video.
    func setupVideoCaptureSession() {
        self.captureSession.beginConfiguration()
        self.previewLayer.videoPreviewLayer.session = self.captureSession
        
        // Add capture session input.
        guard let dev = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            return
        }
        self.cameraDevice = dev
        self.currentCamera = .front
        
        // Add capture session input.
        guard let captureInput = try? AVCaptureDeviceInput(device: self.cameraDevice), self.captureSession.canAddInput(captureInput) else {
            return
        }
        self.captureSession.addInput(captureInput)
        
        // Add capture session output.
        self.videoOutput.setSampleBufferDelegate(self, queue: self.dataOutputQueue)
        self.videoOutput.alwaysDiscardsLateVideoFrames = true
        guard self.captureSession.canAddOutput(self.videoOutput) else {
            return
        }
        
        // Set sRGB as default color space.
        self.captureSession.automaticallyConfiguresCaptureDeviceForWideColor = false
        self.captureSession.sessionPreset = .hd1280x720
        self.videoResolution = CGSize(width: 720, height: 1280)
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
            //self.cameraDevice.activeDepthDataFormat = selectedFormat
            // TODO inspect
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
        
        self.sessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    func stopObservingDevice() {
        self.exposureObservation?.invalidate()
        self.tempObservation?.invalidate()
        self.isoObservation?.invalidate()
    }
    
    func observeDevice() {
        // Start observing camera device exposureDuration.
        self.exposureObservation = observe(\.self.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            DispatchQueue.main.async {
                self.exposureLabel.text = String(Int(1/(newVal.seconds)))
            }
        }
        
        // Start observing camera device white balance gains.
        self.tempObservation = observe(\.self.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            DispatchQueue.main.async {
                self.tempLabel.text = String(Int(temp)) + "K"
            }
        }
        
        // Start observing camera device white balance gains.
        self.isoObservation = observe(\.self.cameraDevice.iso, options: .new){
            obj, change in
            guard let newVal = change.newValue else {
                return
            }
            DispatchQueue.main.async {
                self.isoLabel.text = String(Int(newVal))
            }
        }
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

extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, willCapturePhotoFor resolvedSettings: AVCaptureResolvedPhotoSettings) {
        DispatchQueue.main.async {
            self.previewLayer.videoPreviewLayer.opacity = 0
            UIView.animate(withDuration: 0.25) {
                self.previewLayer.videoPreviewLayer.opacity = 1
            }
        }
    }
    
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("Error in didFinishProcessingPhoto callback: \(error)")
            return
        }
        
        guard let mainCGImage = self.getMainCGImage(photo) else {
            print ("Error getting main CGImage from photo data")
            return
        }
        
        guard let portraitMatte = photo.portraitEffectsMatte else {
            print ("Portait effect Matte not found")
            return
        }
        guard let matteCGImage = self.getPortraitCGImage(portraitMatte.mattingImage, CGSize(width: mainCGImage.width, height: mainCGImage.height)) else {
            print ("Could not convert portrait matte to CGImage")
            return
        }
        let start = CFAbsoluteTimeGetCurrent()
        self.photoProcessor.detectFace(mainCGImage)
        self.photoProcessor.semaphore.wait()
        self.photoProcessor.computeFinalFaceMask(matteCGImage)
        if self.maskSwitch.isOn {
            self.photoProcessor.calculateOverExposedPointsGPU()
        }else {
            self.photoProcessor.calculateOverExposedPointsCPU()
        }
        
        if self.maskSwitch.isOn {
            //self.capturedImage = PhotoProcessor.CGImageToUIImage(mainCGImage)
            self.capturedImage = self.photoProcessor.overlayOverExposedMask()
        }else {
            self.capturedImage = self.photoProcessor.overlayOverExposedMask()
        }
        print ("photo processing time time: \(CFAbsoluteTimeGetCurrent() - start)")
        
        performSegue(withIdentifier: self.imageViewSegue, sender: nil)
    }
    
    // getPortraitCGImage returns portrait CGImage from given portrait pixel buffer and main image resolution.
    func getPortraitCGImage(_ buffer: CVPixelBuffer, _ resolution: CGSize) -> CGImage? {
        var ciImage = CIImage(cvPixelBuffer: buffer).oriented(forExifOrientation: Int32(CGImagePropertyOrientation.right.rawValue))
        
        let scale = CGAffineTransform(scaleX: resolution.width/ciImage.extent.size.width, y: resolution.height/ciImage.extent.size.height)
        ciImage = ciImage.transformed(by: scale)
        
        return PhotoProcessor.CIImageToCGImage(ciImage)
    }
    
    // getMainCGImage returns the main RGB CG image from given AVCapturePhoto.
    func getMainCGImage(_ photo: AVCapturePhoto) -> CGImage? {
        guard let photoData = photo.fileDataRepresentation() else {
            print ("photo Data not found")
            return nil
        }
        
        guard let uiImage = UIImage(data: photoData) else {
            print ("Could not create UI Image from photoData")
            return nil
        }
        
        guard let ciImage = CIImage(image: uiImage)?.oriented(forExifOrientation: Int32(CGImagePropertyOrientation.right.rawValue)) else {
            print ("could not convert uiImage to ciImage")
            return nil
        }
        
        return PhotoProcessor.CIImageToCGImage(ciImage)
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Do nothing for now.
    }
    
    // Stream video frames from camrea input.
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to obtain a CVPixelBuffer for the current output frame.")
            return
        }
        guard let cgImage = PhotoProcessor.CIImageToCGImage(CIImage(cvPixelBuffer: pixelBuffer)) else {
            print ("Could not convert to cgimage")
            return
        }
        let start = CFAbsoluteTimeGetCurrent()
        self.photoProcessor.detectFace(cgImage)
        self.photoProcessor.semaphore.wait()
        DispatchQueue.main.async {
            self.drawFace()
        }
        print ("Processing time: \(CFAbsoluteTimeGetCurrent() - start)")
    }
    
    // setupVisionDrawingLayers sets up overlays for drawing face detection results.
    func setupVisionDrawingLayers() {
        let captureDeviceBounds = CGRect(x: 0,y: 0,width: videoResolution.width,height: videoResolution.height)
        let captureDeviceBoundsCenterPoint = CGPoint(x: captureDeviceBounds.midX, y: captureDeviceBounds.midY)
        let normalizedCenterPoint = CGPoint(x: 0.5, y: 0.5)
        
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"
        overlayLayer.masksToBounds = true
        overlayLayer.anchorPoint = normalizedCenterPoint
        overlayLayer.bounds = captureDeviceBounds
        overlayLayer.position = CGPoint(x: self.previewLayer.videoPreviewLayer.bounds.midX, y: self.previewLayer.videoPreviewLayer.bounds.midY)
        
        let faceRectangleShapeLayer = CAShapeLayer()
        faceRectangleShapeLayer.name = "RectangleOutlineLayer"
        faceRectangleShapeLayer.bounds = captureDeviceBounds
        faceRectangleShapeLayer.anchorPoint = normalizedCenterPoint
        faceRectangleShapeLayer.position = captureDeviceBoundsCenterPoint
        faceRectangleShapeLayer.fillColor = nil
        faceRectangleShapeLayer.strokeColor = UIColor.green.withAlphaComponent(0.7).cgColor
        faceRectangleShapeLayer.lineWidth = 5
        faceRectangleShapeLayer.shadowOpacity = 1.0
        faceRectangleShapeLayer.shadowRadius = 5
        
        let faceLandmarksShapeLayer = CAShapeLayer()
        faceLandmarksShapeLayer.name = "FaceLandmarksLayer"
        faceLandmarksShapeLayer.bounds = captureDeviceBounds
        faceLandmarksShapeLayer.anchorPoint = normalizedCenterPoint
        faceLandmarksShapeLayer.position = captureDeviceBoundsCenterPoint
        faceLandmarksShapeLayer.fillColor = nil
        faceLandmarksShapeLayer.strokeColor = UIColor.yellow.withAlphaComponent(0.7).cgColor
        faceLandmarksShapeLayer.lineWidth = 3
        faceLandmarksShapeLayer.shadowOpacity = 1.0
        faceLandmarksShapeLayer.shadowRadius = 5
        
        faceRectangleShapeLayer.addSublayer(faceLandmarksShapeLayer)
        overlayLayer.addSublayer(faceRectangleShapeLayer)
        
        self.detectionOverlayLayer = overlayLayer
        self.detectedFaceLayer = faceRectangleShapeLayer
        self.detecteddLandMarksLayer = faceLandmarksShapeLayer
        self.previewLayer.videoPreviewLayer.addSublayer(overlayLayer)
        
        self.scaleOverlayGeometry()
    }
    
    // scaleOverlayGeometry scales detection overlay layer to video preview layer coordinates.
    func scaleOverlayGeometry() {
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        let videoPreviewRect = self.previewLayer.videoPreviewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        let affineTransform = CGAffineTransform(scaleX: -videoPreviewRect.width / self.videoResolution.width, y: videoPreviewRect.height/self.videoResolution.height)
        self.detectionOverlayLayer!.setAffineTransform(affineTransform)
        self.detectionOverlayLayer!.position = CGPoint(x: self.previewLayer.bounds.midX, y: self.previewLayer.bounds.midY)
    }
    
    // drawFace draws results from face detection onto video.
    func drawFace() {
        if self.photoProcessor.numFaces == 0 {
            return
        }
        CATransaction.begin()
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectanglePath = CGMutablePath()
        faceRectanglePath.addRect(self.photoProcessor.faceBoundsRect)
        
        /*let landmarksPath = CGMutablePath()
        for p in self.photoProcessor.allFacePoints {
            let rect = CGRect(x: p.x, y: p.y, width: 1, height: 1)
            landmarksPath.addRect(rect)
        }*/
        
        self.detectedFaceLayer!.path = faceRectanglePath
        //self.detecteddLandMarksLayer!.path = landmarksPath
        
        self.scaleOverlayGeometry()
        
        CATransaction.commit()
    }
}

extension ViewController: AVCaptureDataOutputSynchronizerDelegate {
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        
        guard let syncedVideoData = synchronizedDataCollection.synchronizedData(for: self.videoOutput) as? AVCaptureSynchronizedSampleBufferData else {
            return
        }
        guard let syncedDepthData = synchronizedDataCollection.synchronizedData(for: self.depthDataOutput) as? AVCaptureSynchronizedDepthData else {
            return
        }
        if syncedDepthData.depthDataWasDropped || syncedVideoData.sampleBufferWasDropped {
            return
        }
        
        guard let videoPixelBuffer = CMSampleBufferGetImageBuffer(syncedVideoData.sampleBuffer) else {
            print ("Could not convert video sample buffer to cvpixelbuffer")
            return
        }
        
        let depthPixelBuffer = syncedDepthData.depthData.depthDataMap
        guard let mainCGImage = PhotoProcessor.CIImageToCGImage(CIImage(cvPixelBuffer: videoPixelBuffer)) else {
            print ("Could not convert video pixelbuffer to cgimage")
            return
        }
        
        // detect face.
        let start = CFAbsoluteTimeGetCurrent()
        self.photoProcessor.detectFace(mainCGImage)
        
        // draw detected face.
        self.photoProcessor.semaphore.wait()
        if photoProcessor.numFaces == 0 {
            return
        }
        DispatchQueue.main.async {
            self.drawFace()
        }
        self.depthCutOff = computeDepthCutoff(mainCGImage, depthPixelBuffer)
        //convertDepthMapToMask(depthPixelBuffer)
        guard var depthMaskCIImage = convertDepthMapToMask2(CIImage(cvPixelBuffer: depthPixelBuffer)) else {
            print ("Could not convert depth mask to CIImage")
            return
        }
        
        let scale = CGAffineTransform(scaleX:  CGFloat(CVPixelBufferGetWidth(depthPixelBuffer))/CGFloat(mainCGImage.width), y: CGFloat(CVPixelBufferGetHeight(depthPixelBuffer))/CGFloat(mainCGImage.height))
        depthMaskCIImage = depthMaskCIImage.transformed(by: scale)
        
        guard let depthCGImage = PhotoProcessor.CIImageToCGImage(depthMaskCIImage) else {
            print ("cannot convert depth ciimage to cgimage")
            return
        }
        
        self.photoProcessor.computeFinalFaceMask(depthCGImage)
        
        print ("P2 time: \(CFAbsoluteTimeGetCurrent() - start)")
        self.photoProcessor.calculateOverExposedPointsCPU()
        
        
        DispatchQueue.main.async {
            self.drawFace()
        }
        
    }
    
    // computeDepthCutoff returns the depth value at the center of the detected face.
    func computeDepthCutoff(_ mainCGImage: CGImage, _ depthPixelBuffer: CVPixelBuffer) -> Float {
        let scale = CGFloat(CVPixelBufferGetWidth(depthPixelBuffer))/CGFloat(mainCGImage.width)
        guard let face = self.photoProcessor.faceBoundsRect else {
            return -1.0
        }
        let center = CGPoint(x: face.midX, y: face.minY)
        let pixelX = Int((center.x * scale).rounded())
        let pixelY = Int((center.y * scale).rounded())
        
        CVPixelBufferLockBaseAddress(depthPixelBuffer, .readOnly)
        
        let rowData = CVPixelBufferGetBaseAddress(depthPixelBuffer)! + pixelY * CVPixelBufferGetBytesPerRow(depthPixelBuffer)
        let faceCenterDepth = rowData.assumingMemoryBound(to: Float32.self)[pixelX]
        CVPixelBufferUnlockBaseAddress(depthPixelBuffer, .readOnly)
        
        return faceCenterDepth
    }
    
    // convertDepthMapToMask updates given depth map (in place) to a mask using a depth cutoff (already computed).
    // Every pixel above cutoff is converted to 1. otherwise it's 0.
    func convertDepthMapToMask(_ depthPixelBuffer: CVPixelBuffer) {
        let depthWidth = CVPixelBufferGetWidth(depthPixelBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthPixelBuffer)
        
        CVPixelBufferLockBaseAddress(depthPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        for yMap in 0 ..< depthHeight {
            let rowData = CVPixelBufferGetBaseAddress(depthPixelBuffer)! + yMap * CVPixelBufferGetBytesPerRow(depthPixelBuffer)
            let data = UnsafeMutableBufferPointer<Float32>(start: rowData.assumingMemoryBound(to: Float32.self), count: depthWidth)
            for index in 0 ..< depthWidth {
                if data[index] > 0 && data[index] <= self.depthCutOff {
                    data[index] = 1.0
                } else {
                    data[index] = 0.0
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(depthPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
    }
    
    func convertDepthMapToMask2(_ depthMaskCIImage:CIImage) -> CIImage? {
        let s :CGFloat = 4
        let b :CGFloat = 0.25
        
        guard let mat = CIFilter(name: "CIColorMatrix", parameters: ["inputImage": depthMaskCIImage, "inputRVector": CIVector(x: s, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: s, z: 0, w: 0),"inputBVector": CIVector(x: 0, y: 0, z: s, w: 0),"inputBiasVector": CIVector(x: b, y: b, z: b, w: 0)]) else {
            print ("Could not construct CIFilter")
            return nil
        }
        let clamp = CIFilter.colorClamp()
        clamp.inputImage = mat.outputImage
        return clamp.outputImage
    }
}
