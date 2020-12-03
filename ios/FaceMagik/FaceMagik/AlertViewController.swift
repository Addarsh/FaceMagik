//
//  AlertViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/2/20.
//

import UIKit

class AlertViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    
    static func storyboardInstance() -> AlertViewController? {
        let className = String(describing: AlertViewController.self)
        let storyboard = UIStoryboard(name: className, bundle: nil)
        return storyboard.instantiateInitialViewController() as? AlertViewController
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidLoad()
        
        self.startAnimation()
    }
    
    // startAnimation starts animation on given image view.
    func startAnimation() {
        self.imageView.animationImages = self.animatedImages(for: "pitch")
        self.imageView.animationDuration = 2
        self.imageView.animationRepeatCount = 1
        self.imageView.image = self.imageView.animationImages?.last
        self.imageView.startAnimating()
    }
    
    @IBAction func didTapOk() {
        self.dismiss(animated: true)
    }
    
    // animatedImages returns images in given folder name.
    // Image names are assumed to be named 0.png, 1.png ... x.png
    // and in the order of desired animation.
    func animatedImages(for dirName: String) -> [UIImage] {
        var i = 0
        var images = [UIImage]()
        
        while let image = UIImage(named: "\(dirName)/\(i)") {
            images.append(image)
            i += 1
        }
        return images
    }
}
