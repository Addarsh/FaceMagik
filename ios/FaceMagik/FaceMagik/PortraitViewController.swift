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
    @IBOutlet var textView: UITextView!
    @IBOutlet var progressView: CircularProgressView!
    let motionManager = CMMotionManager()
    var motionQueue = OperationQueue()
    let motionFrequency = 1.0/30.0
    let stableOrientationThreshold = 30*3 // 1 seconds at 30 Hz.
    let unstableOrientationThreshold = 30*3 // 1 seconds at 30 Hz.
    let mAvgInterval = 20
    var dataCount: Int = 0
    var pitchArr: [Double]!
    var rollArr: [Double]!
    var yawArr :[Double]!
    var unstableOrientationCount: Int = 0
    var stableOrientationCount: Int = 0
    let htmlInstructions = "Stand up.<br/><br/>Hold up your phone <b>straight</b> in <b>portrait</b> orientation as if you are taking a selfie."
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let notifCenter = NotificationCenter.default
        notifCenter.addObserver(self, selector: #selector(appMovedToBackground), name: UIApplication.didEnterBackgroundNotification, object: nil)
        notifCenter.addObserver(self, selector: #selector(appMovedToForeground), name: UIApplication.willEnterForegroundNotification, object: nil)
        
        self.pitchArr = Array(repeating: 0, count: self.mAvgInterval)
        self.rollArr = Array(repeating: 0, count: self.mAvgInterval)
        self.yawArr = Array(repeating: 0, count: self.mAvgInterval)
        
        DispatchQueue.main.async {
            self.textView.attributedText = self.htmlInstructions.htmlAttributedString()
        }
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
                        self.progressView.animate(0)
                    }
                    self.stableOrientationCount = 0
                    self.unstableOrientationCount += 1
                    if (self.unstableOrientationCount == 1) || (self.unstableOrientationCount % self.unstableOrientationThreshold == 0) {
                        self.presentAlert()
                    }
                    return
                }
                self.unstableOrientationCount = 0
                self.stableOrientationCount += 1
                DispatchQueue.main.async {
                    self.progressView.animate(Float(self.stableOrientationCount)/Float(self.stableOrientationThreshold))
                }
                if self.stableOrientationCount >= self.stableOrientationThreshold {
                    DispatchQueue.main.async {
                        self.checkBox.isEnabled = true
                    }
                }
            }
        })
    }
    
    // presentAlert presents an alert dialog if user is unable to orient the phone correctly.
    func presentAlert() {
        DispatchQueue.main.async {
            if self.presentedViewController != nil {
                // Previous alert is still being presented.
                return
            }
            /*let alert = UIAlertController(title: "Phone orientation incorrect", message: "Please keep the phone straight and in portrait orientation.", preferredStyle: .alert)
            let ok = UIAlertAction(title: "Ok", style: .default , handler: nil)
            alert.addAction(ok)*/
            guard let alert = AlertViewController.storyboardInstance() else {
                return
            }
            self.present(alert, animated: true)
        }
    }
}

extension Sequence where Element: AdditiveArithmetic {
    func sum() -> Element {
        return reduce(.zero, +)
    }
}

extension String {
    func htmlAttributedString() -> NSAttributedString? {
        let htmlTemplate = """
        <!doctype html>
        <html>
          <head>
            <style>
              body {
                font-family: -apple-system;
                font-size: 21px;
              }
            </style>
          </head>
          <body>
            \(self)
          </body>
        </html>
        """

        guard let data = htmlTemplate.data(using: .utf8) else {
            return nil
        }

        guard let attributedString = try? NSAttributedString(
            data: data,
            options: [.documentType: NSAttributedString.DocumentType.html],
            documentAttributes: nil
            ) else {
            return nil
        }

        return attributedString
    }
}
