//
//  RotateInstructionsController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/17/20.
//

import UIKit
import Gifu

class RotateInstructionsController: UIViewController {
    @IBOutlet var imageView: GIFImageView!
    
    static func storyboardInstance() -> RotateInstructionsController? {
        let className = String(describing: RotateInstructionsController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? RotateInstructionsController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.imageView.animate(withGIFNamed: "rotate-woman")
    }
    
    // back allows user to go back to previous veiwcontroller.
    @IBAction func back() {
        self.dismiss(animated: true)
    }
}
