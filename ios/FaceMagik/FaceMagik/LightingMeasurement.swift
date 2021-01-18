//
//  LightingMeasurement.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 1/17/21.
//

import Photos

class LightingMeasurement: NSObject, LightingObserver {
    
    @objc private var cameraDevice: AVCaptureDevice!
    private var exposureObservation: NSKeyValueObservation?
    private var isoObservation: NSKeyValueObservation?
    private var tempObservation: NSKeyValueObservation?
    private var delegate: LightingObserverDelegate?
    
    
    func startObserving(device: AVCaptureDevice?, delegate: LightingObserverDelegate?) {
        guard let dev = device else {
            return
        }
        self.cameraDevice = dev
        self.delegate = delegate
        
        // Find initial values of ISO, exposure and color temperature.
        let currISO = Int(self.cameraDevice.iso)
        let currExposure = Int(1/self.cameraDevice.exposureDuration.seconds)
        let currTemp = Int(self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature)
        
        self.delegate?.updatedISO(iso: currISO)
        self.delegate?.updatedExposure(exposure: currExposure)
        self.delegate?.updatedColorTemp(temp: currTemp)
        
        // Start observing camera device exposureDuration.
        self.exposureObservation = observe(\.self.cameraDevice.exposureDuration, options: .new){
            object, change in
            guard let newVal = change.newValue else {
                return
            }
            self.delegate?.updatedExposure(exposure: Int(1/(newVal.seconds)))
        }
        
        // Start observing camera device ISO.
        self.isoObservation = observe(\.self.cameraDevice.iso, options: .new){
            obj, change in
            guard let newVal = change.newValue else {
                return
            }
            self.delegate?.updatedISO(iso: Int(newVal))
        }
        
        // Start observing camera device white balance gains.
        self.tempObservation = observe(\.self.cameraDevice.deviceWhiteBalanceGains, options: .new){
            obj, chng in
            let temp = self.cameraDevice.temperatureAndTintValues(for: self.cameraDevice.deviceWhiteBalanceGains).temperature
            self.delegate?.updatedColorTemp(temp: Int(temp))
        }
    }
    
    func stopObserving() {
        self.exposureObservation?.invalidate()
        self.isoObservation?.invalidate()
        self.tempObservation?.invalidate()
    }
}
