MAIN TODO

efficient adaptation (adrianna)
[x] current approach: full fine-tuning of all model parameters & only feature extractor is frozen
[x] way to improve: 
	[x] LoRA/QLoRA (train only low rank adapter matrices) -> 90% reduction in trainable parameters
	[x] Adapter tuning (Insert small adapter modules in transformer blocks) -> modular, reusable components
	[x] Knowledge distillation (transfer knowledge from large to smaller model) -> enable edge deployment
	[x] Multilingual Transfer (leverage pre-training on related Bantu languages) -> improved low-resource performace

meaningful evaluation 
[x] test model on a new data (Young)
[x] PER/CER (hyunjoo)
[x] semantic error rate (Young)

responsible practice 

writing up more details in the paper 
[x] preprocessing (Mohamed)
	[x] why we used pydub
	[x] why we used that threshold 
	[x] mainly backing up and reasons (very important!)
[x] data description part (TBD)
	[x] include more description for the data set
	[x] patterns and statistics of the actual/raw data we have
	[x] adding some figures 
	[x] adding tables? 
	[x] how we partitioned dataset
[x] word2vec description (hyunjoo)
	[x] model architecture
	[x] explained why we chose word2vec
	[x] comparison with other model (might just add this other model for the future work section)
		[ ] model A has this architecture, model B has this, blabla, model C is specialised here. therefore, we decided to use...
	[x] specification (--> goes well with table) 
	[x] pipeline (figure, check the citation)
[x] training set up (TBD)
	[x] parameters and such 
[x] limitations (adrianna)
	[x] information of the speaker is not there
	[x] dialectal information is missing
	[x] tones
[x] future work (Mohamed)
