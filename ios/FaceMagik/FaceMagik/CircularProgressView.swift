//
//  CircularProgressView.swift
//  FaceMagik
//
//  Created by Addarsh Chandrasekar on 11/30/20.
//

import UIKit

class CircularProgressView: UIView {
    private var circleLayer = CAShapeLayer()
    private var progressLayer = CAShapeLayer()
    let radius: CGFloat = 40.0
    let startAngle: CGFloat = -CGFloat.pi/2.0
    let endAngle: CGFloat = 3*CGFloat.pi/2.0
    let circLineWidth: CGFloat = 10.0
    let progressLineWidth: CGFloat = 10.0
    let strokeEndKeyPath = "strokeEnd"
    let progressAnimKey = "progressAnim"
    var fromValue: Float = 0
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        self.createLayers()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        
        self.createLayers()
    }
    
    // createLayers initializes circular and progress layer
    // for future animation.
    func createLayers() {
        let circularPath = UIBezierPath(arcCenter: CGPoint(x: frame.width/2.0, y: frame.height/2.0), radius: self.radius, startAngle: self.startAngle, endAngle: self.endAngle, clockwise: true)
        
        self.circleLayer.path = circularPath.cgPath
        self.circleLayer.fillColor = UIColor.clear.cgColor
        self.circleLayer.lineCap = .round
        self.circleLayer.lineWidth = self.circLineWidth
        self.circleLayer.strokeColor = UIColor.white.cgColor
        
        self.progressLayer.path = circularPath.cgPath
        self.progressLayer.fillColor = UIColor.clear.cgColor
        self.progressLayer.lineCap = .round
        self.progressLayer.lineWidth = self.progressLineWidth
        self.progressLayer.strokeColor = UIColor(red: 0.145, green: 0.647, blue: 0.482, alpha: 1.0).cgColor
        self.progressLayer.strokeEnd = 0
        
        self.layer.addSublayer(self.circleLayer)
        self.layer.addSublayer(self.progressLayer)
    }
    
    // animate creates a progress view animation from previous toValue to new toValue.
    func animate(_ toValue: Float) {
        let progressAnimation = CABasicAnimation(keyPath: self.strokeEndKeyPath)
        progressAnimation.fromValue = self.fromValue
        progressAnimation.duration = 0
        progressAnimation.toValue = toValue
        progressAnimation.fillMode = .forwards
        progressAnimation.isRemovedOnCompletion = false
        
        self.progressLayer.add(progressAnimation, forKey: self.progressAnimKey)
        self.fromValue = toValue
    }
    
}
