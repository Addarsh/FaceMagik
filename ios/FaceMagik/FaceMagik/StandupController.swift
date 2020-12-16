//
//  StandupController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/15/20.
//

import UIKit

class StandupController: UIViewController {
    
    static func storyboardInstance() -> StandupController? {
        let className = String(describing: StandupController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? StandupController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    // back allows user to go back to previous veiwcontroller.
    @IBAction func back() {
        self.dismiss(animated: true)
    }
}
