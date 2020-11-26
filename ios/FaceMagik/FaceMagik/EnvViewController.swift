//
//  EnvViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/25/20.
//

import UIKit
import Photos

struct SensorReadout {
    var iso: Int
    var exposure: Int
    var temp: Int
}

class EnvViewController: UIViewController {
    
    @IBOutlet private var previewLayer: PreviewView!
    @IBOutlet var exposureLabel: UILabel!
    @IBOutlet var tempLabel: UILabel!
    @IBOutlet var isoLabel: UILabel!
    @IBOutlet var dirLabel: UILabel!
    @IBOutlet var rotLabel: UILabel!
    
    // AVCaptureSession variables.
    @objc var cameraDevice: AVCaptureDevice!
    var currentCamera: AVCaptureDevice.Position = .unspecified
    var sessionQueue: DispatchQueue!
    var captureSession =  AVCaptureSession()
    var captureOutput: AVCapturePhotoOutput!
    var exposureObservation: NSKeyValueObservation?
    var tempObservation: NSKeyValueObservation?
    var isoObservation: NSKeyValueObservation?
    let segueIdentifier = "videoView"
    var currTemp: Int = 0
    var currISO: Int = 0
    var currExposure: Int = 0
    
    // Location related variables.
    let locationManager = CLLocationManager()
    var sensorDict: [Int: SensorReadout] = [:]
    let totalHeadingVals = 300
    let rotationComplete = "Done"
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.locationManager.delegate = self
        
        self.sessionQueue = DispatchQueue(label: "session queue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem, target: .none)
        
        self.setupPhotoCaptureSession()
        
        self.observeDevice()
        
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
        self.locationManager.startUpdatingHeading()
    }
    
    @objc func appMovedToBackground() {
        if self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.stopRunning()
            }
        }
        self.locationManager.stopUpdatingHeading()
    }
    
    @objc func appMovedToForeground() {
        if !self.captureSession.isRunning {
            DispatchQueue.main.async {
                self.captureSession.startRunning()
            }
        }
        self.locationManager.startUpdatingHeading()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if self.captureSession.isRunning {
            self.captureSession.stopRunning()
        }
        self.locationManager.stopUpdatingHeading()
    }
    
    // setupPhotoCaptureSession sets up a capture session to capture photos.
    func setupPhotoCaptureSession() {
        self.captureSession.beginConfiguration()
        self.previewLayer.videoPreviewLayer.session = self.captureSession
        
        // Add capture session input.
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        self.cameraDevice = dev
        self.currentCamera = .back
        
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
    }
    
    // observeDevice observes the exposure duration and color temperature of current device.
    func observeDevice() {
        // Start observing camera device exposureDuration.
        self.exposureObservation = observe(\.self.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            self.currExposure = Int(1/(newVal.seconds))
            DispatchQueue.main.async {
                self.exposureLabel.text = String(self.currExposure)
            }
        }
        
        // Start observing camera device white balance gains.
        self.tempObservation = observe(\.self.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            self.currTemp = Int(temp)
            DispatchQueue.main.async {
                self.tempLabel.text = String(self.currTemp) + "K"
            }
        }
        
        // Start observing camera device white balance gains.
        self.isoObservation = observe(\.self.cameraDevice.iso, options: .new){
            obj, change in
            guard let newVal = change.newValue else {
                return
            }
            self.currISO = Int(newVal)
            DispatchQueue.main.async {
                self.isoLabel.text = String(self.currISO)
            }
        }
    }
    
    // startVideoProcessing segues into video processing view controller.
    @IBAction func startVideoProcessing() {
        performSegue(withIdentifier: self.segueIdentifier, sender: nil)
    }
    
    
    // switchCamera is a helper function to switch cameras (front to back and vice versa).
    func switchCamera() {
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
    
    // stoObservingDevice stops observing camera device.
    func stopObservingDevice() {
        self.exposureObservation?.invalidate()
        self.tempObservation?.invalidate()
        self.isoObservation?.invalidate()
    }
}

extension EnvViewController: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        DispatchQueue.main.async {
            let heading = Int(newHeading.magneticHeading)
            self.dirLabel.text = String(heading)
            if self.sensorDict[heading] == nil {
                self.sensorDict[heading] = SensorReadout(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp)
            }
            if self.sensorDict.keys.count >= self.totalHeadingVals {
                DispatchQueue.main.async {
                    self.rotLabel.textColor = .systemGreen
                    self.rotLabel.text = String(self.rotationComplete)
                }
            }
        }
    }
}
