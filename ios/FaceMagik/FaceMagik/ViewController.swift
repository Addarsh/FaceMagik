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

    override func viewDidLoad() {
        super.viewDidLoad()
        
        maskSwitch.isOn = false
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)

        sessionQueue = DispatchQueue(label: "session queue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem, target: .none)
        configureCaptureSession()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidLoad()
        if !captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.startRunning()
            }
        }
    }
    
    @objc func appMovedToBackground() {
        if captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.stopRunning()
            }
        }
    }
    
    @objc func appMovedToForeground() {
        if !captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.startRunning()
            }
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    // startPicker starts a UIImagePickerController session.
    @IBAction func startPicker() {
        picker.sourceType = .camera
        picker.cameraDevice = .front
        picker.showsCameraControls = false
        picker.cameraFlashMode = .off
        picker.delegate = self
        picker.cameraOverlayView = overlayView
        
        picker.cameraViewTransform = CGAffineTransform(translationX: 0, y: 120)
        
        present(picker, animated: true)
    }
    
    // dismissPicker dismisses given picker and -re-starts capture session.
    @IBAction func dismissPicker() {
        DispatchQueue.main.async {
            self.captureSession.startRunning()
        }
        picker.dismiss(animated: true)
    }
    
    // Picker take Picture.
    @IBAction func takePicture() {
        picker.takePicture()
    }
    
    // Start recording video.
    @IBAction func startRecording() {
        let photoSettings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
        photoSettings.flashMode = .off
        if currentCamera == .front {
            photoSettings.isDepthDataDeliveryEnabled = true
            photoSettings.isPortraitEffectsMatteDeliveryEnabled = true
        }
        
        captureOutput.capturePhoto(with: photoSettings, delegate: self)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == imageViewSegue {
            guard let destVC = segue.destination as? ImageViewController else {
                return
            }
            destVC.image = capturedImage
            destVC.overExposedPercent = photoProcessor.overExposedPercent()
        }
    }
    
    @IBAction func switchSessions() {
        stopObservingDevice()
        
        guard let currentCameraInput = captureSession.inputs.first else {
            return
        }
        captureSession.beginConfiguration()
        
        captureSession.removeInput(currentCameraInput)
        
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
        cameraDevice = dev
        
        guard let captureInput = try? AVCaptureDeviceInput(device: cameraDevice), captureSession.canAddInput(captureInput) else {
            return
        }
        captureSession.addInput(captureInput)
        
        if newPosition == .front {
            captureOutput.isDepthDataDeliveryEnabled = true
            captureOutput.isPortraitEffectsMatteDeliveryEnabled = true
        } else {
            captureOutput.isDepthDataDeliveryEnabled = false
            captureOutput.isPortraitEffectsMatteDeliveryEnabled = false
        }
        
        if let photoConnection = captureOutput.connection(with: .video) {
            photoConnection.videoOrientation = .portrait
        }
        
        captureSession.commitConfiguration()
        currentCamera = newPosition

        observeDevice()
    }
    
    // configureCaptureSession configures AVCaptureSession.
    func configureCaptureSession() {
        captureSession.beginConfiguration()
        previewLayer.videoPreviewLayer.session = captureSession
        
        // Add capture session input.
        guard let dev = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            return
        }
        cameraDevice = dev
        currentCamera = .front
        
        // Add capture session input.
        guard let captureInput = try? AVCaptureDeviceInput(device: cameraDevice), captureSession.canAddInput(captureInput) else {
            return
        }
        captureSession.addInput(captureInput)
        
        // Add capture session output.
        let photoOutput = AVCapturePhotoOutput()
        guard captureSession.canAddOutput(photoOutput) else {
            return
        }
        
        
        captureSession.sessionPreset = .hd1280x720
        captureSession.addOutput(photoOutput)
            
        if let photoConnection = photoOutput.connection(with: .video) {
            photoConnection.videoOrientation = .portrait
        }
        
        photoOutput.isDepthDataDeliveryEnabled = true
        photoOutput.isPortraitEffectsMatteDeliveryEnabled = true
        captureOutput = photoOutput
        
        captureSession.commitConfiguration()
        
        observeDevice()
        
        sessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    func stopObservingDevice() {
        exposureObservation?.invalidate()
        tempObservation?.invalidate()
        isoObservation?.invalidate()
    }
    
    func observeDevice() {
        // Start observing camera device exposureDuration.
        exposureObservation = observe(\.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            DispatchQueue.main.async {
                self.exposureLabel.text = String(Int(1/(newVal.seconds)))
            }
        }
        
        // Start observing camera device white balance gains.
        tempObservation = observe(\.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            DispatchQueue.main.async {
                self.tempLabel.text = String(Int(temp)) + "K"
            }
        }
        
        // Start observing camera device white balance gains.
        isoObservation = observe(\.cameraDevice.iso, options: .new){
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
        capturedImage = uiimage
        
        performSegue(withIdentifier: imageViewSegue, sender: nil)
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
        
        guard let mainCGImage = getMainCGImage(photo) else {
            print ("Error getting main CGImage from photo data")
            return
        }
        
        guard let portraitMatte = photo.portraitEffectsMatte else {
            print ("Portait effect Matte not found")
            return
        }
        guard let matteCGImage = getPortraitCGImage(portraitMatte.mattingImage, CGSize(width: mainCGImage.width, height: mainCGImage.height)) else {
            print ("Could not convert portrait matte to CGImage")
            return
        }
        let start = CFAbsoluteTimeGetCurrent()
        photoProcessor.prepareDetectionRequest(mainCGImage, matteCGImage)
        photoProcessor.detectFace()
        photoProcessor.semaphore.wait()
        if maskSwitch.isOn {
            capturedImage = PhotoProcessor.CGImageToUIImage(mainCGImage)
        }else {
            capturedImage = photoProcessor.overlayOverExposedMask()
        }
        print ("photo processing time time: \(CFAbsoluteTimeGetCurrent() - start)")
        
        performSegue(withIdentifier: imageViewSegue, sender: nil)
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
