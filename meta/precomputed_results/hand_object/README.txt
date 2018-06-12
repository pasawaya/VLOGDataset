rawPixels.npy       -- raw pixels downsampled then fed to SVM
colorHist.npy       -- color histogram + SVM
r50_midpoint.npy    -- resnet50 on middle frame + SVM
r50_mu.npy          -- resnet50 activations averaged over clip + SVM
r50_ft.npy          -- resnet50 fine-tuned 
i3d_mean.npy        -- i3d average feature + svm
i3d2_mean.npy       -- i3d fine-tuned, then activations + svm (works better)
