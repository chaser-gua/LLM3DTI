
cfg = {
  "drug_in_channels": 1, #100+4096+708,
  "drug_conv1_out_channels": 4,
  "drug_conv1_kernel_size": 256,
  "drug_conv1_stride": 2,
  "drug_conv1_padding": 2,
  "drug_conv2_out_channels": 1,
  "drug_conv2_kernel_size": 128,
  "drug_conv2_stride": 2,
  "drug_conv2_padding": 2,  
  
  "prot_in_channels": 1, #400+4096+1493,
  "prot_conv1_out_channels": 4,
  "prot_conv1_kernel_size": 256,
  "prot_conv1_stride": 2,
  "prot_conv1_padding": 2,  
  "prot_conv2_out_channels": 1,
  "prot_conv2_kernel_size": 128,
  "prot_conv2_stride": 2,
  "prot_conv2_padding": 2,    
  
  "hidden_size": 256,
  "all_head_size": 512,
  "head_num": 4,
  
  # "drug_out_in_features": ,
  "drug_out_out_features": 128,
  # "prot_out_in_features": ,
  "prot_out_out_features": 128,
  
  "label_num": 2,
  
  ###
  "lr": 1e-3,
  "wd": 1e-6,
  "bs": 64,
  
}