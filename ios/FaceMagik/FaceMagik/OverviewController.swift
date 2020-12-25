//
//  OverviewController.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 12/4/20.
//

import UIKit

// Set to true during debug mode/testing.
let testMode: Bool = true

class OverviewController: UIViewController {
    private let pageControl = UIPageControl()
    private let scrollView = UIScrollView()
    @IBOutlet private var button: UIButton!
    let numPages = 3
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.view.addSubview(self.scrollView)
        self.view.addSubview(self.pageControl)
        self.view.addSubview(self.button)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        self.configScrollView()
        self.configPageControl()
    }
    
    // configureScrollView configures scroll view for given view controller.
    func configScrollView() {
        self.scrollView.delegate = self
        self.scrollView.frame = CGRect(x: 0, y: 0, width: self.view.frame.width, height: self.view.frame.height)
        self.scrollView.isPagingEnabled = true
        self.scrollView.contentSize = CGSize(width: self.view.frame.width*CGFloat(self.numPages), height: self.view.frame.height)
        
        let imgList: [String] = ["wand.png", "selfie-woman.png", "makeup-foundation.png"]
        let textList: [String] = ["Welcome to FaceMagik.\n\nMakeup foundations that work for you.", "Take pictures to determine your skin tone.", "Find foundations that match you skin tone."]
        for i in 0..<self.numPages {
            let page = UIView(frame: CGRect(x: CGFloat(i)*self.view.frame.width, y: 0, width: self.view.frame.width, height: self.view.frame.height))
            
            // Add text to page.
            let textView = UITextView(frame: CGRect(x: 50, y: 150, width: 300, height: 400))
            textView.backgroundColor = .white
            textView.textColor = .black
            textView.font = UIFont(name: "Helvetica", size: 26)
            textView.textAlignment = .center
            textView.isEditable = false
            textView.text = textList[i]
            page.addSubview(textView)
            
            // Add image to page.
            let imageView = UIImageView(image: UIImage(named: imgList[i]))
            imageView.frame = CGRect(x: page.frame.width/4, y: 350, width: 200, height: 200)
            page.addSubview(imageView)
            
            self.scrollView.addSubview(page)
        }
    }
    
    // configPageControl conigures page control for given view controller.
    func configPageControl() {
        self.pageControl.numberOfPages = self.numPages
        self.pageControl.pageIndicatorTintColor = .systemGray
        self.pageControl.currentPageIndicatorTintColor = .systemIndigo
        let height :CGFloat = 250
        self.pageControl.frame = CGRect(x: 0, y: self.view.frame.height-height, width: self.view.frame.width, height: height)
    }
    
    // getStarted starts user journey in the app.
    @IBAction func getStarted() {
        if testMode {
            guard let vc = AssessLightController.storyboardInstance() else {
                return
            }
            vc.modalPresentationStyle = .fullScreen
            self.present(vc, animated: true)
        } else {
            guard let vc = LightConditionsController.storyboardInstance() else {
                return
            }
            vc.modalPresentationStyle = .fullScreen
            self.present(vc, animated: true)
        }
    }
    
    @IBAction func unwindToRootViewController(segue: UIStoryboardSegue) {}
}

extension OverviewController: UIScrollViewDelegate {
    func scrollViewDidScroll(_ scrollView: UIScrollView) {
        self.pageControl.currentPage = Int(scrollView.contentOffset.x/scrollView.frame.width)
    }
}
