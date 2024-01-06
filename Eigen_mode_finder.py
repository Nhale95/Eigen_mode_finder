#improt necessary libraries
from tkinter import *
from tkinter.ttk import Scale
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage.segmentation import flood_fill
import scipy
import time
from PIL import ImageTk, Image
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

# Function that takes in  Mtrix to plot
def plot_matrix(matrix,contour,X = [-1,1],i=0,value=0,fractal_order=3):
    #create figure and axis to plot eigenmode in  
    fig,ax = plt.subplots(figsize=(6,6),dpi=200)
    ax.axis("off")
    #Define geometry of axis
    left, bottom, width, height = [0.02, 0.02, 0.96, 0.96]
    ax = fig.add_axes([left, bottom, width, height])
    ax.axis("off")
    #normalise matrix to fit the colorbar
    m = np.max(np.abs(matrix))
    if m < 0:
        m = -1*m
    matrix_n =matrix/m
    #plot matrix and define colorbar and plotting range    
    ax.imshow(matrix_n.transpose()+contour.transpose()*0.1,cmap='RdBu',vmin=-1,vmax = 1)
    return [fig,ax]

# Function which takes in a matrix and solves for the eigenvalues and eigenvectors
def get_eigensystem_spars(coefficient_matrix,k):
    # Compute eigenvalues and eigenvectors, k defines the number of eigenmodes to solve for
    eigval, eigvect = eigs(coefficient_matrix,k,which = "SM",maxiter=50000)
    #take only real components
    eigval = np.real(eigval)
    eigvect = np.real(eigvect)
    # Create ndarray of bools and elimiate trivial eigenvalues
    not_zero = (np.abs(eigval)>0.01)
    eigval = eigval[not_zero] 
    eigvect = eigvect[:,not_zero]
    # Create ndarray of indexes of ordered (ascending) modes
    idx = np.argsort(np.abs(eigval))  
    # reorder eigenvalues 
    eigval = eigval[idx]  # reorder eigenvalues 
    # reorder eigenmodes 
    eigvect = eigvect[:,idx] # reorder eigenmodes 
    #get number of found eigenvalues
    n = len(eigval)
    eigval = np.sqrt(-1*eigval)
    return [eigval,eigvect,n]

# Function that finds the positions of specific values in an array
def get_index_positions(list_of_elems, element):
    #define empy list
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

# Function that takes in a contour matrix z and determines a coefficient matrix which encodes the Helmholts equation boundary condition
def get_sparse_coefficient_matrix_from_shape_2(Z, Lx=1,Ly=1):
    depth = 0
    points = 2
    Nx, Ny = Z.shape
    #Definte lits to include the values of the coefficient matrix
    row = []
    column = []
    # Interval length
    dX = Lx/(Nx-1)
    delta = ((1/4)**depth)*Lx
    data = []
    #Define iterator
    k = 0
    #define empty lists to fill the index of values and their corresponding row and column indeces from the sparse matrix
    Index = []
    Is = []
    Js = []
    # iterate over x values (or columns)
    for i in range(Nx):
        # iterate over y values (or rows)
        for j in range(Ny): 
            index = i*Ny + j
            if Z[i,j] == 2:
                k = k + 1
                Index.append(index)
                Is.append(i)
                Js.append(j)

    Nx, Ny = Z.shape
    #Implementing boundary condition
    # iterate over x values (or columns)
    for ii in range(len(Index)):
        if ii%(int(len(Index)/10))==0:
            print("       = {}".format(np.round(100*ii/len(Index),1)))
        #append results
        row.append(ii)
        column.append(ii)
        data.append(-4/dX**2)
        #repeat process for all boundary conditions of the stencil
        if 0 < Js[ii]:
            if Z[Is[ii],Js[ii]-1]==2:
                row.append(ii-1)
                column.append(ii)
                data.append(1/dX**2)
        if  Js[ii]<Ny -1 :
            if Z[Is[ii],Js[ii]+1]==2:
                row.append(ii+1)
                column.append(ii)
                data.append(1/dX**2)
        if 0 < Is[ii]:
            if Z[Is[ii]-1,Js[ii]]==2:
                set_1 = set(get_index_positions(Is,Is[ii]-1))
                intersection = set_1.intersection(get_index_positions(Js,Js[ii]))
                intersection_as_list = list(intersection)
                row.append(ii)
                column.append(intersection_as_list[0])
                data.append(1/dX**2)
        if Is[ii]<Nx-1:
            if Z[Is[ii]+1,Js[ii]]==2:
                set_1 = set(get_index_positions(Is,Is[ii]+1))
                intersection = set_1.intersection(get_index_positions(Js,Js[ii]))
                intersection_as_list = list(intersection)
                row.append(ii)
                column.append(intersection_as_list[0])
                data.append(1/dX**2)
    #Define Sparse matrix
    M=csr_matrix((data, (row, column)), shape=(k,k))
    #Save values in folder in directory called "Data"
    np.save("Data\X", np.asarray(X))
    np.save("Data\Is", np.asarray(Is))
    np.save("Data\Js", np.asarray(Js))
    scipy.sparse.save_npz('Data\M'.format(depth,points), M)
    return [M,Is,Js]


class paint():
    # Define the constructor method '__init__'
    def __init__(self, root, N,scale):
        # Initialize the object with root window and set its properties
        self.root = root
        self.N = N
        self.root.title("paint")
        self.root.geometry("{}x{}".format(2 * N + 160, N + 60))
        self.root.configure(background="white")
        self.root.resizable(0, 0)
        self.condition = False
        self.scale = scale
        self.increment = 0
        # Create buttons for setting points and making shapes
        self.inside_point_button = Button(self.root, text="set\npoint\ninside", bd=4, bg="white",
                                          command=self.set_condition, width=8, relief=RIDGE)
        self.inside_point_button.place(x=5, y=100)
        self.make_shape_button = Button(self.root, text="\nmake\nshape\n ", bd=4, bg="white",
                                        command=self.make_shape, width=8, relief=RIDGE)
        self.make_shape_button.place(x=5, y=30)

        # Create the Text widget for a text box
        # Create the Text widget for a text box
        self.text_box = Text(self.root, height=2, width=4, wrap="char", bd=4)
        self.text_box.place(x=20, y=170)

        # Insert initial text content into the Text widget
        initial_text_content = "Min\nMode"
        self.text_box.insert("1.0", initial_text_content)

        self.text_entry_min = Entry(self.root, bd=4)
        self.text_entry_min.place(x=5, y=220)

        self.text_entry_max = Entry(self.root, bd=4)
        self.text_entry_max.place(x=5, y=290)
     
        self.text_box_max = Text(self.root, height=2, width=4, wrap="char", bd=4)
        self.text_box_max.place(x=20, y=245)

        # Insert initial text content into the Text widget
        initial_text_content_2 = "Max\nMode"
        self.text_box_max.insert("1.0", initial_text_content_2)

        # Calculate a scaled down value for N
        self.N_ = round(N / self.scale)
        dim = self.N_
        # Initialize a matrix and seed point
        self.matrix = matrix = np.zeros((self.N_, self.N_))
        self.seed_point = (int(self.N_ / 2), int(self.N_ / 2))

        # Create a frame for pen size control
        colors = ["red", "blue"]
        self.pen_size_scale_frame = LabelFrame(self.root, text="size", bd=5, bg="white", font=("ariel", 15, "bold"),
                                               relief=RIDGE)
        self.pen_size_scale_frame.place(x=0, y=440, height=175, width=70)

        # Create a vertical scale for pen size
        self.pen_size = Scale(self.pen_size_scale_frame, orient=VERTICAL, from_=12, to=1, length=150)
        self.pen_size.set(1)
        self.pen_size.grid(row=0, column=1, padx=15)

        # Create two canvases for drawing
        self.canvas = Canvas(self.root, bg="white", bd=5, relief=GROOVE, height=N, width=N)
        self.canvas.place(x=80, y=0)

        self.result_canvas_1 = Canvas(self.root, bg="white", bd=5, relief=GROOVE, height=N, width=N)
        self.result_canvas_1.place(x=N + 89, y=0)
        self.next_button = Button(self.root, text="next", bd=4, bg="white",
                                     command=self.increment_plot, width=8, relief=RIDGE)
        self.next_button.place(x=5, y=350)
        self.back_button = Button(self.root, text="back", bd=4, bg="white",
                                          command=self.back_plot, width=8, relief=RIDGE)
        self.back_button.place(x=5, y=390)
        self.canvas.bind("<B1-Motion>", self.paint)

    # Define the paint method for drawing on the canvas
    def paint(self, event):
        x, y = event.x, event.y
        print(x, y)
        S = self.scale
        # Check the condition for setting a point inside
        if self.condition:
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='blue', outline='blue', width=self.pen_size.get())
            self.seed_point = (round(x / S), round((y) / S))
        else:
            # Draw an oval based on pen size and update the matrix
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='red', outline='red', width=self.pen_size.get())
            try:
                if round(self.pen_size.get()) == 1:
                    self.matrix[round(x / S), round((y) / S)] = 1
                elif 4 >= round(self.pen_size.get()) > 1:
                    self.matrix[round(x / S), round((y) / S)] = 1
                    self.matrix[round(x / S) + 1, round((y) / S)] = 1
                    self.matrix[round(x / S), round((y) / S) + 1] = 1
                elif 8 >= round(self.pen_size.get()) > 4:
                    self.matrix[round(x/S),round((y)/S)] = 1
                    self.matrix[round(x/S)+1,round((y)/S)] = 1
                    self.matrix[round(x/S),round((y)/S)+1] = 1
                    self.matrix[round(x/S)-1,round((y)/S)] = 1
                    self.matrix[round(x/S),round((y)/S)-1] = 1            
                    
                elif round(self.pen_size.get()) > 8:
                    self.matrix[round(x/S),round((y)/S)] = 1
                    self.matrix[round(x/S)+1,round((y)/S)] = 1
                    self.matrix[round(x/S),round((y)/S)+1] = 1
                    self.matrix[round(x/S)-1,round((y)/S)] = 1
                    self.matrix[round(x/S),round((y)/S)-1] = 1     
                    self.matrix[round(x/S)-1,round((y)/S)-1] = 1
                    self.matrix[round(x/S)+1,round((y)/S)-1] = 1
                    self.matrix[round(x/S)-1,round((y)/S)+1] = 1
                    self.matrix[round(x/S)+1,round((y)/S)+1] = 1   
            except:
                pass

    def set_condition(self):
        self.condition = True

    def increment_plot(self):
        self.increment = self.increment + 1
        self.image = PhotoImage(file="figures/mode_{}.png".format(self.increment))  # Replace with your PNG file path

   # Resize the image to fit the canvas
        self.image = self.image.subsample(int(self.image.width() / self.N), int(self.image.height() / self.N))

        # Display the image on the canvas
        self.result_canvas_1.create_image(0, 0, anchor="nw", image=self.image)

        # Redraw the canvas
        self.result_canvas_1.update()

    def back_plot(self):
        self.increment = self.increment - 1
        self.image = PhotoImage(file="figures/mode_{}.png".format(self.increment))  # Replace with your PNG file path

   # Resize the image to fit the canvas
        self.image = self.image.subsample(int(self.image.width() / self.N), int(self.image.height() / self.N))

        # Display the image on the canvas
        self.result_canvas_1.create_image(0, 0, anchor="nw", image=self.image)

        # Redraw the canvas
        self.result_canvas_1.update()

    def make_shape(self):
         matrix_2 = flood_fill(self.matrix, (self.seed_point), 2)
         inside_matrix = matrix_2 - self.matrix
         print(type(inside_matrix))
         print("Size of the matrix:", inside_matrix.shape)

         fig,ax = plt.subplots(figsize=(6,6),dpi=200)
         ax.axis("off")
         left, bottom, width, height = [0.01, 0.01, 0.98, 0.98]
         ax = fig.add_axes([left, bottom, width, height])
         # Define x and y scales
         x_scale = np.arange(0, inside_matrix.shape[1])
         y_scale = np.arange(0, inside_matrix.shape[0])

        # Create meshgrid arrays
         X, Y = np.meshgrid(x_scale, y_scale)
         flipped_matrix = np.flipud(inside_matrix.transpose())

        # Use pcolor to plot the matrix with the desired colormap
         ax.pcolor(X, Y, flipped_matrix, cmap='RdBu', shading='auto')

         fig.savefig("figures/test.png",dpi = 800)

         # Load a PNG image
         self.image = PhotoImage(file="figures/test.png")  # Replace with your PNG file path

         # Resize the image to fit the canvas
         self.image = self.image.subsample(int(self.image.width() / self.N), int(self.image.height() / self.N))

        # Display the image on the canvas
         self.result_canvas_1.create_image(0, 0, anchor="nw", image=self.image)

        # Redraw the canvas
         self.result_canvas_1.update()
         get_sparse_coefficient_matrix_from_shape_2(inside_matrix)
         M = scipy.sparse.load_npz('Data\M.npz')
         Is = np.load("Data\Is.npy")
         Js = np.load("Data\Js.npy")
         l,v,n = get_eigensystem_spars(M,80)
         Nx = self.N_
         grid = np.zeros([Nx,Nx])
         x_scale = np.arange(0, grid.shape[1])
         y_scale = np.arange(0, grid.shape[0])

        ## Create meshgrid arrays
         X, Y = np.meshgrid(x_scale, y_scale)
         mode_min = int(self.text_entry_min.get())
         self.increment = mode_min
         mode_max = int(self.text_entry_max.get())
         for j in range(mode_min,mode_max,1):
            if j < n:
                for i in range(len(Is)):
                    grid[Is[i],Js[i]] = v[:,j][i]
            #np.save("Data\eigen_mode_{}.npy".format(j), grid)
                [fig,ax] = plot_matrix(grid,self.matrix,X=X,i=j,value = l[j],fractal_order = 1)
                ax.text(0.05, 0.95, "{}".format(j), transform=ax.transAxes,fontsize = 18)
                fig.savefig("figures\Mode_{}.png".format(j),dpi = 800)
            ## Remove the current Figure from the canvas
                self.image = PhotoImage(file="figures/Mode_{}.png".format(mode_min))  # Replace with your PNG file path
            # Resize the image to fit the canvas
                self.image = self.image.subsample(int(self.image.width() / self.N), int(self.image.height() / self.N))
            # Display the image on the canvas
                self.result_canvas_1.create_image(0, 0, anchor="nw", image=self.image)
            # Redraw the canvas
                self.result_canvas_1.update()



root = Tk()
p = paint(root,600,3)
root.mainloop()
