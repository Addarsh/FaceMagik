//
//  EnvConditions.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/3/21.
//

import Photos
import CoreMotion

class EnvConditions: NSObject, EnvObserver {
    private var delegate: EnvObserverDelegate?
    @objc private var cameraDevice: AVCaptureDevice!
    private var exposureObservation: NSKeyValueObservation?
    private var isoObservation: NSKeyValueObservation?
    private var tempObservation: NSKeyValueObservation?
    private var currTemp: Int = 0
    private var currISO: Int = 0
    private var currExposure: Int = 0
    private let envQueue = DispatchQueue(label: "Env Sensor Queue", qos: .userInitiated , attributes: [], autoreleaseFrequency: .inherit, target: nil)
    static private let expPercentThreshold = 70
    static private let isoPerentThreshold = 70
    static private let colorTempThreshold = 70
    
    // Core Motion variables.
    private let motionManager = CMMotionManager()
    private var motionQueue = OperationQueue()
    static private let motionFrequency = 1.0/30.0
    static private let totalHeadingVals = 320
    private var sensorMap: [Int: SensorValues] = [:]
    
    func observeLighting(device: AVCaptureDevice, vc: EnvObserverDelegate?) {
        self.cameraDevice = device
        self.delegate = vc
        
        // Start observing camera device exposureDuration.
        self.exposureObservation = observe(\.self.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            self.envQueue.async {
                self.currExposure = Int(1/(newVal.seconds))
            }
            self.delegate?.notifyExposureUpdate(newExpsosure: Int(1/(newVal.seconds)))
        }
        
        // Start observing camera device white balance gains.
        self.isoObservation = observe(\.self.cameraDevice.iso, options: .new){
            obj, change in
            guard let newVal = change.newValue else {
                return
            }
            self.envQueue.async {
                self.currISO = Int(newVal)
            }
            self.delegate?.notifyISOUpdate(newISO: Int(newVal))
        }
        
        // Start observing camera device white balance gains.
        self.tempObservation = observe(\.self.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            self.envQueue.async {
                self.currTemp = Int(temp)
            }
            self.delegate?.notifyTempUpdate(newTemp: Int(temp))
        }
    }
    
    func startMotionUpdates() {
        if !self.motionManager.isDeviceMotionAvailable {
            print ("Device motion unavaible! Error!")
            return
        }
        if self.motionManager.isDeviceMotionActive {
            return
        }
        
        self.motionManager.deviceMotionUpdateInterval = EnvConditions.motionFrequency
        self.motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: self.motionQueue, withHandler: { (data, error) in
            guard let validData = data else {
                return
            }
            let heading = Int(validData.heading)
            self.envQueue.async {
                if self.sensorMap.keys.count >= EnvConditions.totalHeadingVals {
                    // Completed sensor data collection.
                    return
                }
                if self.sensorMap[heading] != nil {
                    return
                }
                self.sensorMap[heading] = SensorValues(iso: self.currISO, exposure: self.currExposure, temp: self.currTemp, sceneType: SceneType.Unknown)
                
                let kCount = self.sensorMap.keys.count
                
                self.delegate?.notifyProgress(progress: Float(kCount)/Float(EnvConditions.totalHeadingVals))
                if kCount == EnvConditions.totalHeadingVals {
                    self.testLighting()
                }
            }
        })
    }
    
    func stopMotionUpdates() {
        if !self.motionManager.isDeviceMotionActive {
            return
        }
        self.motionManager.stopDeviceMotionUpdates()
        self.envQueue.async {
            self.sensorMap = [:]
        }
    }
    
    private func testLighting() {
        var numVisibleColorTemp = 0
        var numGoodISO = 0
        var numGoodExposure = 0
        for (_, readouts) in self.sensorMap {
            if readouts.temp >= 4000 {
                numVisibleColorTemp += 1
            }
            if readouts.iso < 400 {
                numGoodISO += 1
            }
            if readouts.exposure <= 50 {
                numGoodExposure += 1
            }
        }

        let colorTempPercent = Int((Float(numVisibleColorTemp)/Float(self.sensorMap.keys.count))*100.0)
        let isoPercent = Int((Float(numGoodISO)/Float(self.sensorMap.keys.count))*100.0)
        let expPercent = Int((Float(numGoodExposure)/Float(self.sensorMap.keys.count))*100.0)
        
        self.delegate?.notifyLightingTestResults(isIndoors: true, isDayLight: colorTempPercent >= EnvConditions.colorTempThreshold ? true : false, isGoodISO: isoPercent >= EnvConditions.isoPerentThreshold ? true : false, isGoodExposure: expPercent >= EnvConditions.expPercentThreshold ? true : false)
    }
}
