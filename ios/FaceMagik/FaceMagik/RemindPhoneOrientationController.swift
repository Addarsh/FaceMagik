//
//  RemindStandupController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/22/20.
//

import UIKit

class RemindPhoneOrientationController: UIViewController {
    private static let backHomeSegueIdentifier: String = "OverviewController"
    
    static func storyboardInstance() -> RemindPhoneOrientationController? {
        let className = String(describing: RemindPhoneOrientationController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? RemindPhoneOrientationController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // back allows user to go back to root viewcontroller.
    @IBAction func cancel() {
        self.performSegue(withIdentifier: RemindPhoneOrientationController.backHomeSegueIdentifier, sender: self)
    }
    
    // done allows user to go to the next view controller.
    @IBAction func done() {
        guard let vc = AssessFaceController.storyboardInstance() else {
            return
        }
        vc.modalPresentationStyle = .fullScreen
        self.present(vc, animated: true)
    }
}
