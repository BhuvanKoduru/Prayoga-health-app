import tensorflow as tf


def init(): 
	# json_file = open('model.json','r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = tf.keras.models.model_from_json(loaded_model_json)
	#load weights into new model
	model = tf.keras.models.load_model('arth.h5')
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	#model.compile(loss=tf.losses.BinaryCrossentropy(),optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	#graph = tf.get_default_graph()

	return model