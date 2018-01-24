import Foundation

struct Matrix: CustomStringConvertible {
	let rows: Int, columns: Int
	var grid: [Double]
	
	var description: String {
		var thing: String {
			var temp = ""
			for (index, item) in self.grid.enumerated() {
				if (index % self.columns == 0)  {
					temp += "\n\t"
				}
				temp += "\(NSString(format: "%.3f", item)) "
			}
			return temp
		}
		
		return "\n\(rows)r x \(columns)c matrix \(thing)\n"
	}
	
	init(rows: Int, columns: Int) {
		self.rows = rows
		self.columns = columns
		grid = Array(repeating: 0.0, count: rows * columns)
	}
	
	init(rows: Int, columns: Int, data: [Double]) {
		assert(data.count == rows * columns)
		self.rows = rows
		self.columns = columns
		grid = data
	}
	
	mutating func randomize() {
		grid = grid.map { _ in Double.random }
	}
	
	func indexIsValid(row: Int, column: Int) -> Bool {
		return row >= 0 && row < rows && column >= 0 && column < columns
	}
	
	subscript(row: Int, column: Int) -> Double {
		get {
			assert(indexIsValid(row: row, column: column), "Index out of range")
			return grid[(row * columns) + column]
		}
		set {
			assert(indexIsValid(row: row, column: column), "Index out of range")
			grid[(row * columns) + column] = newValue
		}
	}
	
	static func *(first: Matrix, second: Matrix) -> Matrix {
		assert(first.columns == second.rows) // row size == column size
		
		var outputData = Array(repeating: 0.0, count: first.rows * second.columns)
		let outputRows = first.rows
		let outputCols = second.columns
		
		for i in 0..<outputData.count {
			let row = i / (outputCols)
			let col = i % (outputCols)
			
			for j in 0..<first.columns {
				outputData[i] += first[row, j] * second[j, col]
			}
		}
		
		return Matrix(rows: outputRows, columns: outputCols, data: outputData)
	}
	
	static func *(first: Matrix, second: Vector) -> Vector {
		assert(first.columns == second.rows)
		
		let secondToMatrix = Matrix(rows: second.rows, columns: 1, data: second.values)
		let outputMatrix = first * secondToMatrix
		return Vector(rows: first.rows, data: outputMatrix.grid)
	}
	
	static func ==(first: Matrix, second: Matrix) -> Bool {
		return first.grid == second.grid && first.rows == second.rows && first.columns == second.columns
	}
	
	static func ==(first: Matrix, second: Vector) -> Bool {
		return first.grid == second.values && first.rows == second.rows && first.columns == 1
	}
}

struct Vector: CustomStringConvertible {
	let rows: Int
	var values: [Double]
	
	var description: String {
		var thing: String {
			var temp = ""
			for item in self.values {
				temp += "\t\(NSString(format: "%.3f", item))\n"
			}
			return temp
		}
		
		return "\n\(rows)r vector\n\(thing)\n"
	}
	
	init(rows: Int) {
		self.rows = rows
		values = Array(repeating: 0.0, count: rows)
	}
	
	init(rows: Int, data: [Double]) {
		self.rows = rows
		values = data
	}
	
	mutating func randomize() {
		values = values.map { _ in Double.random }
	}
	
	mutating func sigmoid() {
		values = values.map { $0.sigmoid }
	}
	
	func indexIsValid(index: Int) -> Bool {
		return index >= 0 && index < rows
	}
	
	subscript(index: Int) -> Double {
		get {
			assert(indexIsValid(index: index), "Index out of range")
			return values[index]
		}
		set {
			assert(indexIsValid(index: index), "Index out of range")
			values[index] = newValue
		}
	}
	
	static func +(first: Vector, second: Vector) -> Vector {
		assert(first.rows == second.rows)
		
		var output = Vector(rows: first.rows)
		for i in 0..<output.rows {
			output.values[i] = first.values[i] + second.values[i]
		}
		
		return output
	}
	
	static func ==(first: Vector, second: Vector) -> Bool {
		return first.values == second.values && first.rows == second.rows
	}
	
	static func ==(first: Vector, second: Matrix) -> Bool {
		return first.values == second.grid && first.rows == second.rows && second.columns == 1
	}
}



class Network {
	let num_layers: Int
	let sizes: [Int]
	var weights: [Matrix] = []
	var biases: [Vector] = []
	var layers: [Vector] = []

    init(sizes: [Int]) {
        num_layers = sizes.count
        self.sizes = sizes
        weights = initWeights()
		biases = initBiases()
		layers = initLayers()
    }
	
	func initWeights() -> [Matrix] {
		var array: [Matrix] = []
		
		for size_index in 0..<num_layers - 1 {
			var newMatrix = Matrix(rows: sizes[size_index + 1], columns: sizes[size_index])
			newMatrix.randomize()
			array.append(newMatrix)
		}

		return array
	}
	
	func initBiases() -> [Vector] {
		var array: [Vector] = []
		
		for size_index in 1..<num_layers {
			var newVector = Vector(rows: sizes[size_index])
			newVector.randomize()
			array.append(newVector)
		}
		
		return array
	}
	
	func initLayers() -> [Vector] {
		var array: [Vector] = []
		for size_index in 0..<num_layers {
			array.append(Vector(rows: sizes[size_index]))
		}
		
		return array
	}
	
	func feedForward(input: [Double]) -> [Double] {
		assert(input.count == sizes[0])
		assert((input.filter{$0 > 1.0 || $0 < 0.0}).count == 0)
		
		layers[0] = Vector(rows: sizes[0], data: input)
		print("Set input data")
		
		for layerIndex in 0..<num_layers - 1 {
			print("\nStarting to compute layer \(layerIndex + 1)")
			print("Weights: \(weights[layerIndex])")
			print("Layer: \(layers[layerIndex])")

			let multiplicationResult = weights[layerIndex] * layers[layerIndex]
			var addBias = multiplicationResult + biases[layerIndex]
			addBias.sigmoid()

			layers[layerIndex + 1] = addBias
		}

		return layers[num_layers - 1].values
	}
}

extension Double {
    public static var random: Double {
        return Double(arc4random()) / Double(UINT32_MAX)
    }
    
    public static func random(min: Double, max: Double) -> Double {
        return Double.random * (max-min) + min
    }
	
	public var sigmoid: Double {
		return 1.0/(1.0 + exp(self * -1))
	}
}

// UNIT TESTS

assert(
	Matrix(rows: 3, columns: 3, data: [1, 2, 3, 4, 5, 6, 7, 8, 9])
		* Vector(rows: 3, data: [0, 1, 0])
		== Vector(rows: 3, data: [2, 5, 8]),
	"Unit Test Failed")

assert(
	Matrix(rows: 2, columns: 3, data: [1, 2, 3, 4, 5, 6])
		* Matrix(rows: 3, columns: 2, data: [7, 8, 9, 10, 11, 12])
		== Matrix(rows: 2, columns: 2, data: [58, 64, 139, 154]),
	"Unit Test Failed")



// FEED FORWARD PROGRAM

var network = Network(sizes: [6, 3, 3, 1])
let result = network.feedForward(input: [0.9, 0.8, 0.6, 0.3, 0.1, 0.1])
print("Result:\n\(result)")