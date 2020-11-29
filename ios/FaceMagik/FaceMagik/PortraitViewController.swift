//
//  PortraitViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/28/20.
//

import UIKit
import CoreMotion

class PortraitViewController: UIViewController {
    private let segueIdentifier = "envView"
    @IBOutlet var checkBox: UIButton!
    let motionManager = CMMotionManager()
    var motionQueue = OperationQueue()
    let motionFrequency = 1.0/30.0
    let stableIterationsThreshold = 30*3 // 3 seconds at 30 Hz.
    let mAvgInterval = 20
    var dataCount: Int = 0
    var pitchArr: [Double]!
    var rollArr: [Double]!
    var yawArr :[Double]!
    var stableIterationsCount: Int = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.pitchArr = Array(repeating: 0, count: self.mAvgInterval)
        self.rollArr = Array(repeating: 0, count: self.mAvgInterval)
        self.yawArr = Array(repeating: 0, count: self.mAvgInterval)
    }
    
    
    // goToNextView segues to next view in storyboard.
    @IBAction func goToNextView() {
        performSegue(withIdentifier: self.segueIdentifier, sender: nil)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        super.prepare(for: segue, sender: sender)

        if let destVC = segue.destination as? EnvViewController {
            destVC.modalPresentationStyle = .fullScreen
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidLoad()
        if self.motionManager.isDeviceMotionAvailable {
            self.startMotionUpdates()
        }
    }
    
    @objc func appMovedToBackground() {
        if self.motionManager.isDeviceMotionAvailable {
            self.motionManager.stopDeviceMotionUpdates()
        }
    }
    
    @objc func appMovedToForeground() {
        if self.motionManager.isDeviceMotionAvailable {
            self.startMotionUpdates()
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        if self.motionManager.isDeviceMotionAvailable {
            self.motionManager.stopDeviceMotionUpdates()
        }
    }
    
    // getMovingAverages computes and retrieves moving average values for roll, pitch and yaw
    // from given device motion data.
    func getMovingAverages(data: CMDeviceMotion) -> (pitch: Double, roll: Double, yaw: Double) {
        let idx = self.dataCount % self.mAvgInterval
        
        self.pitchArr[idx] = data.attitude.pitch
        self.rollArr[idx] = data.attitude.roll
        self.yawArr[idx] = data.attitude.yaw
        
        self.dataCount += 1
        
        let pitchAvg = self.pitchArr.sum()/Double(self.dataCount <= self.mAvgInterval ? idx : self.mAvgInterval)
        let rollAvg = self.rollArr.sum()/Double(self.dataCount <= self.mAvgInterval ? idx : self.mAvgInterval)
        let yawAvg = self.yawArr.sum()/Double(self.dataCount <= self.mAvgInterval ? idx : self.mAvgInterval)
        
        return (pitchAvg, rollAvg, yawAvg)
    }
    
    // startMotionUpdates starts to receive motion updates from motion manager.
    func startMotionUpdates() {
        self.motionManager.deviceMotionUpdateInterval = self.motionFrequency
        self.motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: self.motionQueue, withHandler: { (data, error) in
            if let validData = data {
                // For pitch, valid values are between 1.1 and 1.5.
                // For roll, valid values are between -0.3 to 0.4.
                // For yaw, valid values are between 1 to 3.
                let result = self.getMovingAverages(data:validData)
                if (result.pitch < 1.1 || result.pitch > 1.5) || (result.roll < -0.3 || result.roll > 0.4 ) ||
                    (result.yaw < 1 || result.yaw > 3) {
                    DispatchQueue.main.async {
                        self.checkBox.isEnabled = false
                    }
                    self.stableIterationsCount = 0
                    return
                }
                self.stableIterationsCount += 1
                if self.stableIterationsCount >= self.stableIterationsCount {
                    DispatchQueue.main.async {
                        self.checkBox.isEnabled = true
                    }
                }
            }
        })
    }
}

extension Sequence where Element: AdditiveArithmetic {
    func sum() -> Element {
        return reduce(.zero, +)
    }
}
