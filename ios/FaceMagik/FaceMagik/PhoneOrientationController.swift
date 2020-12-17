//
//  PhoneOrientationController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/16/20.
//

import UIKit

class PhoneOrientationController: UIViewController {
    static func storyboardInstance() -> PhoneOrientationController? {
        let className = String(describing: PhoneOrientationController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? PhoneOrientationController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // back allows user to go back to previous veiwcontroller.
    @IBAction func back() {
        self.dismiss(animated: true)
    }
    
    // done allows user to progress to next view controller.
    @IBAction func done() {
        guard let vc = RotateInstructionsController.storyboardInstance() else {
            return
        }
        vc.modalPresentationStyle = .fullScreen
        self.present(vc, animated: true)
    }
}
