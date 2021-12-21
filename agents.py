import random
import math



BOT_NAME = "The Uncontested" #+ 19 


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """
            Determine the minimax utility value of the given state.
            Args:
                state: a connect383.GameState object representing the current board
            Returns: 
                the exact minimax utility value of the state
        """
        if state.is_full():
            return state.utility()
        
        if state.next_player()==1:
            maxScore = -math.inf
            stateSuccessors = state.successors()
            for _, tuple in enumerate(stateSuccessors):
                #tuple[1] = the state
                currState = tuple[1]
                currScore = self.minimax(currState)
                maxScore = max(maxScore, currScore)
            return maxScore

        if state.next_player()==-1:
            minScore = +math.inf
            stateSuccessors = state.successors()
            for _, tuple in enumerate(stateSuccessors):
                currState = tuple[1]
                currScore = self.minimax(currState)
                minScore = min(minScore, currScore)
            return minScore


class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimaxDepth(self, state, depth):
        if state.is_full() or depth == 0:
            return self.evaluation(state)
        
        if state.next_player()==1:
            maxScore = -math.inf
            stateSuccessors = state.successors()
            for _, tuple in enumerate(stateSuccessors):
                #tuple[1] = the state
                currState = tuple[1] #successor state
                currScore = self.minimaxDepth(currState, depth-1) #score of successor state
                maxScore = max(maxScore, currScore)
            return maxScore
        
        if state.next_player()==-1:
            minScore = +math.inf
            stateSuccessors = state.successors()
            for _, tuple in enumerate(stateSuccessors):
                currState = tuple[1]
                currScore = self.minimaxDepth(currState, depth-1)
                minScore = min(minScore, currScore)
            return minScore
        
    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        return self.minimaxDepth(state, self.depth_limit)           

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in constant time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        p1_score = 0
        p2_score = 0
        
        for run in state.get_rows() + state.get_cols() + state.get_diags():
            for elt, length in streaks(run):
                if (elt == 1) and (length==2): #max player
                    for i in run:
                        if (i < len(run)-2):
                            if (run[i+1]==1 and run[i+2]==1 and (run[i]==None or run[i]==1)):
                                if i+3 < len(run) and (run[i+3] == None or run[i+3] == 1):
                                    p1_score+=2
                                else:
                                    p1_score+=1 
                            if (run[i]==1 and run[i+1]==1 and (run[i+2] == None or run[i+2] == 1)): 
                                p1_score+=1
                elif (elt == -1) and (length==2): #min player
                    for i in run:
                        if (i < len(run)-2):
                            if (run[i+1]==-1 and run[i+2]==-1 and (run[i]==None or run[i]==-1)):
                                if i+3 < len(run) and (run[i+3] == None or run[i+3] == -1):
                                    p2_score+=2
                                else:
                                    p2_score+=1 
                            if (run[i]==-1 and run[i+1]==-1 and (run[i+2] == None or run[i+2] == -1)): 
                                p2_score+=1
        
        return p1_score - p2_score

def streaks(lst):  
    """Get the lengths of all the streaks of the same element in a sequence."""
    rets = []  # list of (element, length) tuples
    prev = lst[0]
    curr_len = 1
    for curr in lst[1:]:
        if curr == prev:
            curr_len += 1
        else:
            rets.append((prev, curr_len))
            prev = curr
            curr_len = 1
    rets.append((prev, curr_len))
    return rets

class MinimaxPruneAgent(MinimaxAgent):
    def alphaBetaPruning(self, state, alpha, beta):
            if state.is_full():
                return state.utility()
            
            if state.next_player()==1:
                maxScore = -math.inf
                stateSuccessors = state.successors()
                for _, tuple in enumerate(stateSuccessors):
                    #tuple[1] = the state
                    currState = tuple[1]
                    currScore = self.alphaBetaPruning(currState, alpha, beta)
                    maxScore = max(maxScore, currScore)
                    alpha = max(alpha, currScore)
                    if alpha >= beta:
                        break
                return maxScore

            if state.next_player()==-1:
                minScore = +math.inf
                stateSuccessors = state.successors()
                for _, tuple in enumerate(stateSuccessors):
                    currState = tuple[1]
                    currScore = self.alphaBetaPruning(currState, alpha, beta)
                    minScore = min(minScore, currScore)
                    beta = min(beta, currScore)
                    if alpha >= beta:
                        break
                return minScore
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""
    def minimax(self, state):
        
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not use a depth limit like MinimaxHeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        return self.alphaBetaPruning(state, -math.inf, +math.inf)


# N.B.: The following class is provided for convenience only; you do not need to implement it!

class OtherMinimaxHeuristicAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 26  # Change this line, unless you have something better to do.