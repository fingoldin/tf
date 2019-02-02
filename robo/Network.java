import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.lang.Math;

public class Network {
	int numLayers;
	static int layerOneSize;
	private class Layer{
		double[][] weights;
		double[] biases;
		int neuro;
		int prevNeuro;
		public Layer(int numNeurons, int prevNeurons){
			System.out.println("Created " + numNeurons + " " + prevNeurons);
			weights = new double[prevNeurons][numNeurons];
			neuro = numNeurons;
			prevNeuro = prevNeurons;
			biases = new double[numNeurons];
		}
		public void setWeight(int neuron, int prevNeuron, double weight){
			if(neuron >= 0 && neuron < neuro && prevNeuron >= 0 && prevNeuron < prevNeuro){
				weights[prevNeuron][neuron] = weight;
			}
		}
		public void setBias(int neuron, double bias){
			if(neuron >= 0 && neuron < neuro){
				 biases[neuron] = bias;
			}
		}
		public void printWeights(){
			for (int y = 0; y < neuro; y++){
				for (int z = 0; z < prevNeuro; z++) {
					System.out.println("Weight for " + y + ", " + z + " is " + weights[y][z]);
				}
			}
		}
		public void printBiases(){
			for (int y = 0; y < neuro; y++){
					System.out.println("Bias for " + y + " is " + biases[y]);
			}
		}
		public double[] feed(double[] prevOutputs) {
			double[] outputs = new double[neuro];
			if (prevOutputs.length == prevNeuro) {
				for (int y = 0; y < neuro; y++) {
					double sum = 0.0;
					for (int z = 0; z < prevNeuro; z++) {
						sum += prevOutputs[z] * weights[z][y];
					}
					sum += biases[y];
					outputs[y] = Math.tanh(sum);
				}
			} else {
				System.out.println("Can we get an F in the chat");
			}
			return outputs;
		}
		public int getNumNeurons(){
			return neuro;
		}
		public int getPrevNeurons(){
			return prevNeuro;
		}
	}
	private ArrayList<Layer> layers;
	public Network(File f) {
		layers = new ArrayList<Layer>();
		try {		
			Scanner s = new Scanner(f);
			numLayers = s.nextInt();
			//System.out.println("Number of Layers: " + numLayers);
			int prevLayerSize = s.nextInt();
			layerOneSize = prevLayerSize;
			//System.out.println("Layer 1 Size: " + prevLayerSize);
			for (int x = 1; x < numLayers; x++){
				int layerSize = s.nextInt();
				Layer l = new Layer(layerSize, prevLayerSize);
				//System.out.println("Layer " + (x+1) + " Size: " + layerSize);
				
				for (int z = 0; z < l.getPrevNeurons(); z++){
					for (int y = 0; y < l.getNumNeurons(); y++){
						double weight = s.nextDouble();
						l.setWeight(y, z, weight);
						if (x == 1) {
							System.out.println("Weight " + y + ", " + z + ": " + weight);
						}
					}
					
				}
				for (int y = 0; y < l.getNumNeurons(); y++){
					double bias = s.nextDouble();
					l.setBias(y, bias);
					//System.out.println("Layer " + (x+1) + " Neuron " + y + " Bias: " + bias);
				}
				layers.add(l);
				prevLayerSize = layerSize;
				
			}
			s.close();
		} catch (Exception e){
			System.out.println("Network Construction Error: " + e);
		}
	}
	public Layer getLayer(int layerNumber){
		//System.out.println("Returning Layer " + layerNumber);
		return layers.get(layerNumber);
	}
	public double[] feed(double[] inputs){
		double[] prevOutputs = inputs.clone();
		for (int y = 0; y < numLayers-1; y++) {
			prevOutputs = layers.get(y).feed(prevOutputs);
		}
		return prevOutputs;
	}
	public static void main (String[] args) {
		Network n = new Network(new File("./data.text"));
		double[] inputs = {0.0, -2.0, 0.0, 0.0};
		double outputs[] = n.feed(inputs);
		for (int i = 0; i < outputs.length; i++) {
			System.out.println(i + ": " + (0.5 * (1.0 + outputs[i])));
		}
	}
}
