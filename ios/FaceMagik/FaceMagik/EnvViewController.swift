//
//  EnvViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/25/20.
//

import UIKit
import Photos

class EnvViewController: UIViewController {
    
    @IBOutlet private var previewLayer: PreviewView!
    @IBOutlet var exposureLabel: UILabel!
    @IBOutlet var tempLabel: UILabel!
    @IBOutlet var isoLabel: UILabel!
    @IBOutlet var dirLabel: UILabel!
    
    // AVCaptureSession variables.
    @objc var cameraDevice: AVCaptureDevice!
    var sessionQueue: DispatchQueue!
    var captureSession =  AVCaptureSession()
    var captureOutput: AVCapturePhotoOutput!
    var exposureObservation: NSKeyValueObservation?
    var tempObservation: NSKeyValueObservation?
    var isoObservation: NSKeyValueObservation?
    var segueIdentifier = "videoView"
    
    var locationManager = CLLocationManager()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.locationManager.delegate = self
        
        self.sessionQueue = DispatchQueue(label: "session queue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem, target: .none)
        
        self.setupPhotoCaptureSession()
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
        
        self.observeDevice()
        
        self.sessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    // observeDevice observes the exposure duration and color temperature of current device.
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
    
    // startVideoProcessing segues into video processing view controller.
    @IBAction func startVideoProcessing() {
        performSegue(withIdentifier: self.segueIdentifier, sender: nil)
    }
}

extension EnvViewController: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        DispatchQueue.main.async {
            self.dirLabel.text = String(Int(newHeading.magneticHeading))
        }
    }
}
