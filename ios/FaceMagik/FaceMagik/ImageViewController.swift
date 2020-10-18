//
//  ImageViewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 10/23/20.
//
import UIKit

class ImageViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    var image: UIImage!

    override func viewDidLoad() {
        super.viewDidLoad()
        
        imageView.image = image
    }
}
