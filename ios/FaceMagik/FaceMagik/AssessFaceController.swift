//
//  AssessFaceController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/25/20.
//

import UIKit

class AssessFaceController: UIViewController {
    
    static func storyboardInstance() -> AssessFaceController? {
        let className = String(describing: AssessFaceController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? AssessFaceController
    }
    
    
    // back allowes user to go back to previous view controller.
    @IBAction func back() {
        self.dismiss(animated: true)
    }
}
