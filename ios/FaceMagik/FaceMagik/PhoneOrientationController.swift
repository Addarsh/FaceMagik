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
}
