
#include <queue>

//https://www.geeksforgeeks.org/linked-list-set-1-introduction/

template<class A, class B, class C, class...Z>
class Node
{
public:
  
  Node()=default;
  virtual ~Node()=default;
  
  A mx0;
  B mx1;
  C mx2;
  
  Node(const A& a, const B& b, const C& c) : mx0(a), mx1(b), mx2(c) {}
    
  Node* next;
};

/* Given a reference (pointer to pointer)  
to the head of a list and data,  
inserts a new node on the front of the list. */
//template <typename elem1, typename elem2>
template<class A, class B, class...Z>
void push(Node<A,B,Z...>** head_ref, A new_data1, B new_data2)  
{  
    /* 1. allocate node */
    //Node<elem1,elem2>* new_node = new Node<elem1,elem2>();
    auto new_node = new Node<A,B,Z...>();  
  
    /* 2. put in the data */
    new_node->mx0 = new_data1;
    new_node->mx1 = new_data2;  
    
    /* 3. Make next of new node as head */
    new_node->next = (*head_ref);  
  
    /* 4. move the head to point to the new node */
    (*head_ref) = new_node;  
}  

/* Given a node prev_node, insert a new node after the given 
   prev_node */
template<class A, class B, class...Z>
void insertAfter(Node<A,B,Z...>* prev_node, A new_data) 
{ 
    /*1. check if the given prev_node is NULL */ 
    if (prev_node == NULL)  
    {  
       printf("the given previous node cannot be NULL");        
       return;   
    }   
           
    /* 2. allocate new node */
    auto new_node = new Node<A,B,Z...>(); 
   
    /* 3. put in the data  */
    new_node->mx0 = new_data; 
   
    /* 4. Make next of new node as next of prev_node */
    new_node->next = prev_node->next;  
   
    /* 5. move the next of prev_node as new_node */
    prev_node->next = new_node; 
} 

// This function prints contents of linked list 
// starting from the given node 
template<class A, class B, class...Z>
void printList(Node<A,B,Z...>* n) 
{ 
    while (n != NULL) { 
        std::cout << n->data << " "; 
        n = n->next; 
    } 
} 


template<class A, class B, class...Z>
void Linked_List_introduction() 
{ 
  auto head = new Node<A,B,Z...>(33,0,7);
  auto second = new Node<A,B,Z...>();
  auto third = new Node<A,B,Z...>();
    
    /* Three blocks have been allocated dynamically.  
    We have pointers to these three blocks as head,  
    second and third      
    head         second         third  
        |             |             |  
        |             |             |  
    +---+-----+     +----+----+     +----+----+  
    | # | # |     | # | # |     | # | # |  
    +---+-----+     +----+----+     +----+----+  
      
# represents any random value.  
Data is random because we haven’t assigned  
anything yet */
  
   // head->mx0 = 4; // assign data in first node 
    head->mx1 = 2; // assign data in first node 
    head->next = second; // Link first node with 
    // the second node 
  
    /* data has been assigned to the data part of first  
    block (block pointed by the head). And next  
    pointer of the first block points to second.  
    So they both are linked.  
  
    head         second         third  
        |             |             |  
        |             |             |  
    +---+---+     +----+----+     +-----+----+  
    | 1 | o----->| # | # |     | # | # |  
    +---+---+     +----+----+     +-----+----+      
*/
  
    // assign data to second node 
    second->mx0 = 3; 
    second->mx1 = 4; 
  
    // Link second node with the third node 
    second->next = third; 
  
    /* data has been assigned to the data part of the second  
    block (block pointed by second). And next  
    pointer of the second block points to the third  
    block. So all three blocks are linked.  
      
    head         second         third  
        |             |             |  
        |             |             |  
    +---+---+     +---+---+     +----+----+  
    | 1 | o----->| 2 | o-----> | # | # |  
    +---+---+     +---+---+     +----+----+     */
  
    third->mx0 = 5;  // assign data to third node 
    third->mx1 = 6;  // assign data to third node 
    third->next = NULL; 
  
    /* data has been assigned to the data part of the third  
    block (block pointed by third). And next pointer  
    of the third block is made NULL to indicate  
    that the linked list is terminated here.  
  
    We have the linked list ready.  
  
        head      
            |  
            |  
        +---+---+     +---+---+     +----+------+  
        | 1 | o----->| 2 | o-----> | 3 | NULL |  
        +---+---+     +---+---+     +----+------+      
      
      
    Note that only the head is sufficient to represent  
    the whole list. We can traverse the complete  
    list by following the next pointers. */
  
  push(&head, {.2,0}, {.2,0});
  std:: cout << head->mx0 + head->next->mx0 << std::endl;
  
  std:: cout << std::endl;
  std::cout << head->next->mx2 << std::endl;
  
} 

///////////////////////

//https://www.geeksforgeeks.org/given-linked-list-representation-of-complete-tree-convert-it-to-linked-representation/

// Linked list node 
template<class A, class B, class...Z>
class ListNode
{
public:
  
  ListNode()=default;
  virtual ~ListNode()=default;
  
  A mx0;
  B mx1;
  
  ListNode* next;
};
  
// Binary tree node class 
template<class A, class B, class...Z>
class BinaryTreeNode
{
public:
  
  BinaryTreeNode()=default;
  virtual ~BinaryTreeNode()=default;
  
  A mx0;
  B mx1;
  
  BinaryTreeNode *left, *right; 
 
 int key; 
 
};

// Function to insert a node at the beginning of the Linked List
template<class A, class B, class...Z>
void push(ListNode<A,B,Z...>** head_ref, A new_data) 
{ 
    // allocate node and assign data 
    auto new_node = new ListNode<A,B,Z...>(); 
    new_node->mx0 = new_data; 
  
    // link the old list off the new node 
    new_node->next = (*head_ref); 
  
    // move the head to point to the new node 
    (*head_ref)    = new_node; 
} 
  
// method to create a new binary tree node from the given data 
template<class A, class B, class...Z>
BinaryTreeNode<A,B,Z...>* newBinaryTreeNode(A data) 
{ 
    auto temp = new BinaryTreeNode<A,B,Z...>(); 
    temp->mx0 = data; 
    temp->left = temp->right = NULL; 
    return temp; 
} 
  
// converts a given linked list representing a complete binary tree into the 
// linked representation of binary tree.
template<class A, class B, class...Z>
void convertList2Binary(ListNode<A,B,Z...> *head, BinaryTreeNode<A,B,Z...>* &root) 
{ 
    // queue to store the parent nodes 
    std::queue<BinaryTreeNode<A,B,Z...> *> q; 
  
    // Base Case 
    if (head == NULL) 
    { 
        root = NULL; // Note that root is passed by reference 
        return; 
    } 
  
    // 1.) The first node is always the root node, and add it to the queue 
    root = newBinaryTreeNode<A,B,Z...>(head->mx0); 
    q.push(root); 
  
    // advance the pointer to the next node 
    head = head->next; 
  
    // until the end of linked list is reached, do the following steps 
    while (head) 
    { 
        // 2.a) take the parent node from the q and remove it from q 
        auto parent = q.front(); 
        q.pop(); 
  
        // 2.c) take next two nodes from the linked list. We will add 
        // them as children of the current parent node in step 2.b. Push them 
        // into the queue so that they will be parents to the future nodes 
        BinaryTreeNode<A,B,Z...> *leftChild = NULL, *rightChild = NULL; 
        leftChild = newBinaryTreeNode<A,B,Z...>(head->mx0); 
        q.push(leftChild); 
        head = head->next; 
        if (head) 
        { 
            rightChild = newBinaryTreeNode<A,B,Z...>(head->mx0); 
            q.push(rightChild); 
            head = head->next; 
        } 
  
        // 2.b) assign the left and right children of parent 
        parent->left = leftChild; 
        parent->right = rightChild; 
    } 
} 
  
// Utility function to traverse the binary tree after conversion 
template<class A, class B, class...Z> 
void inorderTraversal(BinaryTreeNode<A,B,Z...>* root) 
{ 
    if (root) 
    { 
        inorderTraversal( root->left ); 
        std::cout << root->mx0 << " "; 
        inorderTraversal( root->right ); 
    } 
} 

/* Computes the number of nodes in a tree. */
template<class A, class B, class...Z>
int size(BinaryTreeNode<A,B,Z...>* node)  
{  
    if (node == NULL)  
        return 0;  
    else
        return(size(node->left) + 1 + size(node->right));  
}  

template<class A, class B, class...Z>
void Construct_Complete_Binary_Tree_from_its_Linked_List_Representation()
{ 
    // create a linked list shown in above diagram 
    class ListNode<A,B,Z...>* head = NULL; 
    push(&head, {36,0});  /* Last node of Linked List */
    push(&head, {30,0}); 
    push(&head, {25,0}); 
    push(&head, {15,0}); 
    push(&head, {12,0}); 
    push(&head, {10,0}); /* First node of Linked List */
  
    BinaryTreeNode<A,B,Z...> *root; 
    convertList2Binary(head, root); 
  
    std::cout << "\nInorder Traversal of the constructed Binary Tree is: \n"; 
    inorderTraversal(root);

    //Inorder Traversal of the constructed Binary Tree is:
    //25 12 30 10 36 15
    
    std::cout << "\nSize of the constructed Binary Tree is: " << size(root) << "\n"; 
} 


////////////////

//https://www.geeksforgeeks.org/flatten-a-binary-tree-into-linked-list/  
/* utility that allocates a new Node  
   with the given key  */
template<class A, class B, class...Z>
BinaryTreeNode<A,B,Z...>* newNode(int key) 
{ 
    auto node = new BinaryTreeNode<A,B,Z...>; 
    node->key = key; 
    node->left = node->right = NULL; 
    return (node); 
} 
  
// Function to convert binary tree into 
// linked list by altering the right node 
// and making left node point to NULL
template<class A, class B, class...Z>  
void flatten(BinaryTreeNode<A,B,Z...>* root) 
{ 
    // base condition- return if root is NULL 
    // or if it is a leaf node 
    if ((root == NULL || root->left == NULL) && 
                        root->right == NULL) { 
        return; 
    } 
  
    // if root->left exists then we have  
    // to make it root->right 
    if (root->left != NULL) { 
  
        // move left recursively 
        flatten(root->left); 
     
        // store the node root->right 
        class BinaryTreeNode<A,B,Z...>* tmpRight = root->right; 
        root->right = root->left; 
        root->left = NULL; 
  
        // find the position to insert 
        // the stored value    
        class BinaryTreeNode<A,B,Z...>* t = root->right; 
        while (t->right != NULL) { 
            t = t->right; 
        } 
  
        // insert the stored value 
        t->right = tmpRight; 
    } 
  
    // now call the same function 
    // for root->right 
    flatten(root->right); 
} 
  
// To find the inorder traversal
template<class A, class B, class...Z>  
void inorder(BinaryTreeNode<A,B,Z...>* root) 
{ 
    // base condition 
    if (root == NULL) 
        return; 
    inorder(root->left); 
    std::cout << root->key << " "; 
    inorder(root->right); 
} 

template<class A, class B, class...Z> 
void flatten_binary_tree() 
{ 
    /*    1 
        /   \ 
       2     5 
      / \     \ 
     3   4     6 */
    BinaryTreeNode<A,B,Z...>* root = newNode<A,B,Z...>(1); 
    root->left = newNode<A,B,Z...>(2); 
    root->right = newNode<A,B,Z...>(5); 
    root->left->left = newNode<A,B,Z...>(3); 
    root->left->right = newNode<A,B,Z...>(4); 
    root->right->right = newNode<A,B,Z...>(6); 
  
    flatten(root); 
  
    std::cout << "The Inorder traversal after "
            "flattening binary tree "; 
    inorder(root);  
} 


/// weighted directed graph
namespace wdg
{

// Data structure to store Adjacency list nodes
template<class A, class B, class C, class...Z>
class Node
{
public:
  Node()=default;
  virtual ~Node()=default;
  
  int val, cost;
	Node* next;
};

// Data structure to store graph edges
template<class A, class B, class C, class...Z>
class Edge
{
public:
  Edge()=default;
  virtual ~Edge()=default;
  int src, dest, weight;
};

template<class A, class B, class C, class...Z>
class Graph
{
	// Function to allocate new node of Adjacency List
	Node<A,B,Z...>* getAdjListNode(int value, int weight, Node<A,B,Z...>* head)
	{
		Node<A,B,Z...>* newNode = new Node<A,B,Z...>;
		newNode->val = value;
		newNode->cost = weight;

		// point new node to current head
		newNode->next = head;

		return newNode;
	}

	int N;	// number of nodes in the graph

public:

	// An array of pointers to Node to represent
	// adjacency list
	Node<A,B,Z...> **head;

	// Constructor
	Graph(Edge<A,B,Z...> edges[], int n, int N)
	{
		// allocate memory
		head = new Node<A,B,Z...>*[N]();
		this->N = N;

		// initialize head pointer for all vertices
		for (int i = 0; i < N; ++i)
			head[i] = nullptr;

		// add edges to the directed graph
		for (unsigned i = 0; i < n; i++)
		{
			int src = edges[i].src;
			int dest = edges[i].dest;
			int weight = edges[i].weight;

			// insert in the beginning
			Node<A,B,Z...>* newNode = getAdjListNode(dest, weight, head[src]);

			// point head pointer to new node
			head[src] = newNode;

			// Uncomment below lines for undirected graph

			/*
			newNode = getAdjListNode(src, weight, head[dest]);

			// change head pointer to point to the new node
			head[dest] = newNode;
			*/
		}
	}

	// Destructor
	virtual ~Graph() {
		for (int i = 0; i < N; i++)
			delete[] head[i];

		delete[] head;
	}
};

// print all neighboring vertices of given vertex
template<class A, class B, class C, class...Z>
void printList(Node<A,B,Z...>* ptr, int i)
{
	while (ptr != nullptr)
	{
		std::cout << "(" << i << ", " << ptr->val
			<< ", " << ptr->cost << ") ";

		ptr = ptr->next;
	}

	std::cout << std::endl;
}

// Graph Implementation in C++ without using STL
template<class A, class B, class C, class...Z>
int driver()
{
	// array of graph edges as per above diagram.
	Edge<A,B,Z...> edges[] =
	{
		// (x, y, w) -> edge from x to y having weight w
		{ 0, 1, 6 }, { 1, 2, 7 }, { 2, 0, 5 }, { 2, 1, 4 },
		{ 3, 2, 10 }, { 4, 5, 1 }, { 5, 4, 3 }
	};

	// Number of vertices in the graph
	int N = 6;

	// calculate number of edges
	int n = sizeof(edges)/sizeof(edges[0]);

	// construct graph
	Graph<A,B,Z...> graph(edges, n, N);

	// print adjacency list representation of graph
	for (int i = 0; i < N; i++)
	{
		// print all neighboring vertices of vertex i
		printList(graph.head[i], i);
	}

	return 0;
}

}