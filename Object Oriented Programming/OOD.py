class Dog():  # class name follow Camel casing

    # class object attribute
    # same for any instance of a class
    species = 'mammal'

    # init method
    # which is going to be called upon whenever you
    # actually create an instance of the class.
    def __init__(self, mybreed, name, spots):
        # attributes
        self.breed = mybreed
        self.name = name
        # expect boolean True/False
        self.spots = spots

    # first method
    # Methods are essentially functions defined inside the body of the class
    # Perform operations that sometimes utilize the actual attributes of the object we created.
    def bark(self):
        print('Woof')

my_sample = SampleWord() # instance of the class
my_dog = Dog()
