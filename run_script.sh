python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 4 4 -embedding_size 4 -int_mode  
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 8 8 -embedding_size 4 -int_mode  
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 16 16 -embedding_size 4 -int_mode  
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 32 32 -embedding_size 4 -int_mode  
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 64 64 -embedding_size 4 -int_mode  
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 128 128 -embedding_size 4 -int_mode  
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 4 4 -embedding_size 4 -int_mode -no_mode 
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 8 8 -embedding_size 4 -int_mode -no_mode
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 16 16 -embedding_size 4 -int_mode -no_mode
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 32 32 -embedding_size 4 -int_mode -no_mode
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 64 64 -embedding_size 4 -int_mode -no_mode
python train_simplenn.py -trn_site Site_25 -test_site Site_1 -mlp_layer 128 128 -embedding_size 4 -int_mode -no_mode

python train.py -rnn_cell gru -rnn_layer 64 64 -output_layer 64 64 -trn_site Site_25 -test_site Site_1 -share_encoder -int_mode
python train.py -rnn_cell gru -rnn_layer 64 64 -output_layer 64 64 -trn_site Site_25 -test_site Site_1 -share_encoder -int_mode -no_mode
