import os
import matplotlib.pyplot as plt
from collections import defaultdict,deque
import heapq
import math
from os.path import splitext
from queue import PriorityQueue
script_path = os.path.dirname(os.path.realpath(__file__))
filenames = []
outpath = []
#subdirs = ["input/level_1/","input/level_2/"]
for path, subdirs, files in os.walk(script_path):
    #print(path,subdirs)
    for name in files:
        if splitext(name)[1].lower() == ".txt":
            filepath = os.path.join(path, name)
            print(os.path.join(path, name))
            filenames.append(filepath)
            abc = filepath.replace('input','output',1)
            abc = abc.replace('.txt','')
            outpath.append(abc)
"""
Tham khảo:
- https://github.com/VaibhavSaini19/A_Star-algorithm-in-Python/blob/master/A-Star%20Algorithm
"""
with open('maze_map_01.txt', 'w') as outfile:
  outfile.write('2\n')
  outfile.write('3 6 -3\n')
  outfile.write('5 14 -1\n')
  outfile.write('xxxxxxxxxxxxxxxxxxxxxx\n')
  outfile.write('x   x   xx xx        x\n')
  outfile.write('x     x     xxxxxxxxxx\n')
  outfile.write('x x    xx  xxxx xxx xx\n')
  outfile.write('      x x xx   xxxx  x\n')
  outfile.write('x          xx  xx  x x\n')
  outfile.write('xxxxxxx x       x  x x\n')
  outfile.write('xxxxxxxxx  x x  xx   x\n')
  outfile.write('x          x x Sx x  x\n')
  outfile.write('xxxxx x  x x x     x x\n')
  outfile.write('xxxxxxxxxxxxxxxxxxxxxx')

with open('maze_map_02.txt', 'w') as outfile:
    outfile.write('0\n')
    outfile.write('xxxxxxxxxxxxx xxxxxxxxxxxxx\n')
    outfile.write('x     x       x     x   x x\n')
    outfile.write('xxx xxxxxxx x x xxxxxxx x x\n')
    outfile.write('x   x       x x x x x     x\n')
    outfile.write('xxx xxx xxx xxx x x x x xxx\n')
    outfile.write('x         x   x   x   x   x\n')
    outfile.write('x xxx xxxxx xxxxx x xxxxx x\n')
    outfile.write('x x x x       x       x   x\n')
    outfile.write('x x xxx x xxx xxxxx x xxx x\n')
    outfile.write('x     x x x     x   x   x x\n')
    outfile.write('x xxxxxxxxx xxx x xxxxxxxxx\n')
    outfile.write('x   x x   x   x x   x     x\n')
    outfile.write('x x x x xxxxx xxx xxx xxxxx\n')
    outfile.write('x x         xS            x\n')
    outfile.write('xxxxxxxxxxxxxxxxxxxxxxxxxxx\n')

with open('maze_map_03.txt', 'w') as outfile:
    outfile.write('0\n')
    outfile.write('x xxxxxxxxxxxxxxxxxxxxxxxxx\n')
    outfile.write('x x x       x x x   x   x x\n')
    outfile.write('x x x xxxxxxx x x xxx x x x\n')
    outfile.write('x     x x     x   x   x   x\n')
    outfile.write('x xxxxx xxx x x x xxxxxxx x\n')
    outfile.write('x     x     x x x x       x\n')
    outfile.write('x xxx xxx xxx x x xxx x xxx\n')
    outfile.write('x   x x     x S x     x   x\n')
    outfile.write('x x x x xxxxxxx x xxx x x x\n')
    outfile.write('x x x x x       x   x x x x\n')
    outfile.write('xxxxx xxxxxxx xxx xxxxx xxx\n')
    outfile.write('x             x     x     x\n')
    outfile.write('x xxxxxxx xxxxxxxxx xxx xxx\n')
    outfile.write('x   x         x       x   x\n')
    outfile.write('xxxxxxxxxxxxxxxxxxxxxxxxxxx\n')

with open('maze_map_04.txt', 'w') as outfile:
    outfile.write('0\n')
    outfile.write('xxxxxxxxxxxxxxxxxxxxxxxxxxx\n')
    outfile.write('  x   x         x         x\n')
    outfile.write('x x xxx x xxx x x x xxxxx x\n')
    outfile.write('x x   x x x x x   x x     x\n')
    outfile.write('x xxx x xxx xxx xxx x x xxx\n')
    outfile.write('x x x x     x x x   x x x x\n')
    outfile.write('x x x x xxxxx x x xxx x x x\n')
    outfile.write('x   x   x       x x   x x x\n')
    outfile.write('x x x x xxx x xxx x x xxx x\n')
    outfile.write('x x   x x   x   x x x x   x\n')
    outfile.write('xxxxxxxxx x x x x xSx x xxx\n')
    outfile.write('x x       x x x x x x     x\n')
    outfile.write('x x xxxxx xxx xxx x x xxx x\n')
    outfile.write('x       x   x x     x x   x\n')
    outfile.write('xxxxxxxxxxxxxxxxxxxxxxxxxxx\n')

with open('maze_map_05.txt', 'w') as outfile:
    outfile.write('0\n')
    outfile.write('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')
    outfile.write('x                                 x\n')
    outfile.write('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxx\n')
    outfile.write('x                                 x\n')
    outfile.write('x xxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxx\n')
    outfile.write('x         x                 x     x\n')
    outfile.write('x x xxxxxxxxxxxxx xxxxxxxxx x xxxxx\n')
    outfile.write('x x x   x   x   x x   x   x x      \n')
    outfile.write('x x x x x x x x x x x x x x x xxxxx\n')
    outfile.write('x x x x x x x x x x x x x x x     x\n')
    outfile.write('x x x x x x x x x x x   x   x x x x\n')
    outfile.write('x x x x x x x x x xxxxxxxxxxx x x x\n')
    outfile.write('x x S x   x   x               x x x\n')
    outfile.write('x xxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxx\n')
    outfile.write('x                                 x\n')
    outfile.write('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')

class Node:
    def __init__(self,x,y, par = None , d = 0):
        self.X = x
        self.Y = y
        self.par = par
        self.d = d
    def display(self):
        print('(' + str(self.X) + ',' + str(self.Y) + '): ' + str(self.d))
    def __lt__(self, other):
        return self.d < other.d
    def __eq__(self,G):
        return self.X == G.X and self.Y == G.Y
    def LEFT(self,G):
        return self.X == G.X and self.Y - G.Y == 1
    def RIGHT(self,G):
        return self.X == G.X and self.Y - G.Y == -1
    def UP(self,G):
        return self.X - G.X == -1 and self.Y == G.Y
    def DOWN(self,G):
        return self.X - G.X == 1 and self.Y == G.Y 

def checkInArray(temp, Arr):
    for x in Arr:
        if x.__eq__(temp):
            return True
    return False

def path(O,S):
    solve = []
    while True:
        if O.__eq__(S):
            break
        p = (O.X , O.Y)
        solve.append(p)
        O = O.par
    return solve

def BFS(route1, route2, solveRoute):
    Open = []
    Closed = []
    Value = 0
    End = Node(end[0],end[1])
    Start = Node(start[0],start[1])

    Open.append(Start)
    
    while True:
        if len(Open) == 0:
            print('tim kiem that bai')
            break
        O = Open.pop(0)
        Closed.append(O)
    
        if O.__eq__(End):
            print('tim thay')
            solveRoute = path(O,Start).copy()
            Value = len(solveRoute)
            break
        lst = []
        for i in route1:
            tmp = Node(i[0],i[1])
            if (O.LEFT(tmp) or O.RIGHT(tmp) or O.UP(tmp) or O.DOWN(tmp)) and not tmp.__eq__(Start):
                lst.append(tmp)
        for x in lst:
            temp = x
            temp.par = O
            ok1 = checkInArray(temp, Open)
            ok2 = checkInArray(temp, Closed)
            if not ok1 and not ok2:
                Open.append(temp)
                temp = (temp.X,temp.Y)
                route2.append(temp)
    return solveRoute,Value

def DFS(route1, route2, solveRoute):
    Open = []
    Closed = []
    Value = 0
    End = Node(end[0],end[1])
    Start = Node(start[0],start[1])

    Open.append(Start)
    
    while True:
        if len(Open) == 0:
            print('tim kiem that bai')
            break
        O = Open.pop(0)
        Closed.append(O)
    
        if O.__eq__(End):
            print('tim thay')
            solveRoute = path(O,Start).copy()
            Value = len(solveRoute)
            break
        lst = []
        for i in route1:
            tmp = Node(i[0],i[1])
            if (O.LEFT(tmp) or O.RIGHT(tmp) or O.UP(tmp) or O.DOWN(tmp)) and not tmp.__eq__(Start) :
                lst.append(tmp)
        pos = 0
        for x in lst:
            temp = x
            temp.par = O
            ok1 = checkInArray(temp, Open)
            ok2 = checkInArray(temp, Closed)
            if not ok1 and not ok2:
                Open.append(temp)
                Open.insert(pos,temp)
                pos += 1
                temp = (temp.X,temp.Y)
                route2.append(temp)
    return solveRoute,Value 

def UCS(route1, route2, solveRoute):
    Open = []
    Closed = []
    Value = 0
    End = Node(end[0],end[1])
    Start = Node(start[0],start[1])
    Open.append(Start)
    
    while True:
        if len(Open) == 0:
            print('tim kiem that bai')
            break

        #Kiem tra trong hang doi Open:----
        '''
        for i in range(len(Open)):
            print('(' + str(Open[i].X) + ',' + str(Open[i].Y) + '): ' + str(Open[i].d) + '|',end=' ')
        print('\n')
        '''   
        #--------------------

        O = Open.pop(0)
        
        Closed.append(O)

        if O.__eq__(End):
            print('tim thay')
            solveRoute = path(O,Start).copy()
            Value = len(solveRoute)
            break
        lst = []
        for i in route1:
            tmp = Node(i[0],i[1])
            if (O.LEFT(tmp) or O.RIGHT(tmp) or O.UP(tmp) or O.DOWN(tmp)) and not tmp.__eq__(Start):
                lst.append(tmp)
        
        for x in lst:
            temp = x
            temp.par = O
            temp.d = O.d + 1

            ok1 = checkInArray(temp, Open)
            ok2 = checkInArray(temp, Closed)

            if not ok1 and not ok2:
                Open.append(temp)
                tmp = (temp.X,temp.Y)
                route2.append(tmp)
        if len(Open) != 0:
            Open.sort()
    return solveRoute,Value

def visualize_maze(matrix, bonus, start, end, algorithm,route,outpath):
    """
     algo.append("BFS")
    algo.append("DFS")
    algo.append("UCS")
    algo.append("ASTAR1")
    algo.append("ASTAR2")
    algo.append("GBFS1")
    algo.append("GBFS2")
    """
    folderAlgo = {
      "BFS" : "bfs",
      "DFS" : "dfs",
      "UCS" : "ucs",
      "ASTAR1" : "astar",
      "ASTAR2" : "astar",
      "GBFS1" : "gbfs",
      "GBFS2" : "gbfs",
    }
    nameAlgo = {
      "BFS" : "bfs",
      "DFS" : "dfs",
      "UCS" : "ucs",
      "ASTAR1" : "astar_heuristic_1",
      "ASTAR2" : "astar_heuristic_2",
      "GBFS1" : "gbfs_heuristic_1",
      "GBFS2" : "gbfs_heuristic_2",
    }
    print(" Dang chay thuat toan: ",algorithm)
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0]))if matrix[i][j]=='x']

    #(i != end[0] or j != end[1])
    route1 = []
    
    for i in range(len(matrix)): 
        for j in range(len(matrix[0])):
            if matrix[i][j] == ' ' or (i == start[0] and j == start[1]):
                route1.append((i,j))
    
    route2 = []
    solve = []


    #2: Select the algorithm
    cost1 = 0
    if algorithm == 'BFS':
        solve,cost = BFS(route1,route2,solve)
        print("cost: ",cost)
        cost1 = cost 
    elif algorithm == 'DFS':
        solve,cost = DFS(route1,route2,solve)
        print("cost: ",cost)
        cost1 = cost 
    elif algorithm == 'UCS':
        solve,cost = UCS(route1,route2,solve)
        print("cost: ",cost)
        cost1 = cost 
    elif algorithm == 'ASTAR1':
        adj = Graph(convertMatrixToWeightedGraph(matrix))
        solve,cost = adj.astar(start,end,Manhattan_distance,bonus)
        print("cost: ",cost)
        cost1 = cost 
    elif algorithm == 'ASTAR2':
        adj = Graph(convertMatrixToWeightedGraph(matrix))
        solve,cost = adj.astar(start,end,Euclidean_distance,bonus)
        print("cost: ",cost)
        cost1 = cost 
    elif algorithm == 'GBFS1':
        adj = Graph(convertMatrixToWeightedGraph(matrix))
        solve,cost = adj.gbfs(start,end,Euclidean_distance,bonus)
        print("cost: ",cost)
        cost1 = cost 
    elif algorithm == 'GBFS2':
        adj = Graph(convertMatrixToWeightedGraph(matrix))
        solve,cost = adj.gbfs(start,end,Diagonal_distance,bonus)
        print("cost: ",cost)
        cost1 = cost 

    #3. Drawing the map
    ax=plt.figure(dpi=100).add_subplot(111)
    for i in ['top','bottom','right','left']:
        ax.spines['top'].set_visible(False)
    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green')
    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')

    if route2:
        for i in range(len(route2)):
            plt.scatter(route2[i][1],-route2[i][0],
                        marker='8',color='silver')
    if solve:
        for i in range(len(solve)):
            plt.scatter(solve[i][1],-solve[i][0],
                        marker='8',color='blue')
    plt.text(end[1],-end[0],'EXIT',color='red',
         horizontalalignment='center',
         verticalalignment='center')

    plt.xticks([])
    plt.yticks([])
    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')
    for _, point in enumerate(bonus):
      print(f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')
    
    #save file
    print(outpath,folderAlgo[algorithm])
    filepath = os.path.join(outpath,folderAlgo[algorithm])
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    namepath1 = nameAlgo[algorithm] + ".txt"
    namepath2 = nameAlgo[algorithm] + ".png"
    fpath1 = os.path.join(filepath,namepath1)
    
    fpath2 = os.path.join(filepath,namepath2)
    print(fpath1,fpath2)
    with open(fpath1,'w') as outfile:
        outfile.write(str(cost1))
    plt.savefig(fpath2)
    plt.show()
    

def read_file(file_name: str = 'maze.txt'):
  f=open(file_name,'r')
  n_bonus_points = int(next(f)[:-1])
  bonus_points = []
  for i in range(n_bonus_points):
    x, y, reward = map(int, next(f)[:-1].split(' '))
    bonus_points.append((x, y, reward))
  text=f.read()
  matrix=[list(i) for i in text.splitlines()]
  f.close()
  return bonus_points, matrix
def printMatrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(matrix[i][j],end='')
        print('')
# tham khảo : https://stackabuse.com/courses/graphs-in-python-theory-and-implementation/lessons/a-star-search-algorithm/
class Graph:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

  def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

  def get_neighbors(self, v):
        return self.adjacency_list[v]
  def astar(self,start_node,stop_node,heuristic,bonus_points = []):
        print(start_node,stop_node)
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        #print(start_node,stop_node)
        open_list = set([start_node])
        closed_list = set([])
        bp_dict = dict()
        for x, y, b in bonus_points:
          if (x, y) not in bp_dict:
            bp_dict[(x, y)] = b
        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = defaultdict(list)
    
        g[start_node] = 0
        f = defaultdict(list)
        f[start_node] = 0
        # parents contains an adjacency map of all nodes
        parents = defaultdict(list)
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None
            print("open list: ",open_list)
            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + heuristic(v,stop_node) < g[n] + heuristic(n,stop_node):
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None
           # print(n,end=': ')
            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                value = f[n]
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path,value
            #print(self.get_neighbors(n),end = ' ')
            # for all neighbors of the current node do
            for (x,y) in self.get_neighbors(n):
                print("neighbor: ",x,y)
                weight = 1
                m = (x,y)
               # print("test",m,weight,end = " ")
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                    if m in bp_dict:
                        g[m] = g[m] - 1 + bp_dict[m]
                    f[m] = g[m] + heuristic(m,stop_node)
                    print("g f ",g[m],f[m])
                   # print(m,end = ' ' )

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        if m in bp_dict:
                           g[m] = g[m] - 1 + bp_dict[m]
                        f[m] = g[m] + heuristic(m,stop_node)
                        print("g f ",g[m],f[m])
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
                            #print(m,end = ' ' )

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)
            #print(' ' )

        print('Path does not exist!')
        return None
  def gbfs(self,start_node,stop_node,heuristic,bonus_points = None):
    waiting = PriorityQueue()
    waiting.put((0, start))
    explored = []
    
    bp_dict = dict()
    for x, y, b in bonus_points:
        if (x, y) not in bp_dict:
            bp_dict[(x, y)] = b
    
    parent,f = {}, {}
    parent[start_node], f[start_node] = None, 0

    while not waiting.empty():
        current = waiting.get()[1]

        if (current == stop_node):
            route = []
            check = end
            cost_val = 0
            total_bonuses = 0

            while check != start:
                route.append(check)
                cost_val += 1
                
                if check in bp_dict:
                    total_bonuses += bp_dict[check]
                    cost_val += bp_dict[check]
                    print(check)
                
                check = parent[check]
                
            cost_val = 0 if cost_val < 0 else cost_val
            print('Total bonuses:', total_bonuses)
            print('Total cost:', cost_val)

            route.append(start_node)
            route.reverse()
            
            return route, cost_val
        
        for (x,y) in self.get_neighbors(current):
            point = (x,y)
            print("point",point)
            
            next_cost = f[current]
            
            if point in bp_dict:
              next_cost += bp_dict[point]
            
            next_cost += 1

            if ((point not in f) or (next_cost < f[point])) and (point not in explored):
                f[point] = next_cost
                waiting.put((heuristic(point,stop_node), point))
                explored.append(point)
                parent[point] = current

    return None,None
def convertMatrixToUnweightedGraph(mat):
    rows = len(mat)
    cols = len(mat[0])
    graph = defaultdict(list)
    for x in range(rows):
        for y in range(cols):
            if mat[x][y] == ' ':
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if 0 <= x+dx < rows and 0 <= y+dy < cols and mat[x+dx][y+dy] == ' ':
                        graph[(x, y)].append([(x+dx, y+dy),1])
    return graph
def convertMatrixToWeightedGraph(mat):
    rows = len(mat)
    cols = len(mat[0])
    graph = defaultdict(list)
    for x in range(rows):
        for y in range(cols):
            if mat[x][y] == ' ' or mat[x][y] == 'S':
                for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                    if 0 <= x+dx < rows and 0 <= y+dy < cols and mat[x+dx][y+dy] == ' ':
                        #lst1 = [x+dx,y+dy]
                        #lst2 = [lst1,1]
                        graph[(x, y)].append((x+dx,y+dy))
                        #print(graph[(x,y)])
    return graph

def Manhattan_distance(x,y):
    dx = abs (x[0] - y[0]) 
    dy = abs (x[1]- y[1]) 
    return (dx + dy)
def Euclidean_distance(x,y):
    dx = abs (x[0] - y[0]) 
    dy = abs (x[1]- y[1]) 
    return abs(math.sqrt(dx*dx + dy*dy))
def Diagonal_distance(x,y):
    dx = abs (x[0] - y[0]) 
    dy = abs (x[1]- y[1]) 
    return dx + dy - min(dx,dy)
def astar(graph,start,goal,heuristic):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()
        print("current: ",current)
        if current == goal:
            break
        print("graph[current]: ",graph[current])
        for next in graph[current]:
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far
        
#----------------------------------- RUN -------------------------------------
for i in range(0,len(filenames)):
    filename = filenames[i]
    outname = outpath[i]
    print("out: ",outname)
    print("Dang chay file input: ",filename)
    bonus_points, matrix = read_file(filename)

    print(f'The height of the matrix: {len(matrix)}')
    print(f'The width of the matrix: {len(matrix[0])}')
    for i in range(len(matrix)):
     for j in range(len(matrix[0])):
        if matrix[i][j]=='S':
            start=(i,j)
        elif matrix[i][j]==' ':
            if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                end=(i,j)
        else:
            pass
    algo = []
    algo.append("BFS")
    algo.append("DFS")
    algo.append("UCS")
    algo.append("ASTAR1")
    algo.append("ASTAR2")
    algo.append("GBFS1")
    algo.append("GBFS2")

    for i in range(0,len(algo)):
        visualize_maze(matrix,bonus_points,start,end,algo[i],None,outname)

    printMatrix(matrix)
    adj = Graph(convertMatrixToWeightedGraph(matrix))
    adj.astar(start,end,Manhattan_distance)


    print(adj[(8,15)])
    print(astar(adj,start,end,'Manhattan_distance'))


    print(adj)

    abc = adj.adjacency_list
    for i in abc:
        print(i,end = ': ')
        print(type(i))
        print(i[0],i[1])
        print('')

visualize_maze(matrix,bonus_points,start,end,'DFS')
