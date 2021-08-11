#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

ll isPrime(ll n) {
    if(n<=1) return 0;
    for (ll i = 2; i*i <= n; ++i) if(n%i==0) return 0;
    return 1;
}

void sieve(ll Prime[], ll primeMark[], ll nPrime, ll n) {
    ll i, j, limit = sqrt(n*1.)+2;
    primeMark[1]=1;
    for(i=4; i<=n; i+=2) primeMark[i]=1;
    Prime[nPrime++]=2;
    for(i=3; i<=3; i+=2) if (!primeMark[i]) {
            Prime[nPrime++]=i;
            if (i <= limit) for (j = i * i; j <= n; j += i * 2)
                    primeMark[j]=1;
        }
}

void Divisors(vector<int> divisors[], int n){
    for(int i = 1; i <= n; i++) for(int j = i; j <= n; j += i)
        divisors[j].push_back(i);
}

int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
int lcm(int a, int b) { return a * (b / gcd(a, b)); }

void sievePhi(int phi[], int phiMark[], int n) {
    for(int i=1; i<=n; i++) phi[i]=i;
    phi[1]=1;
    phiMark[1]=1;
    for (int i = 2; i <= n; i++) if (!phiMark[i]) for (int j = i; j <= n; j += i)
        phiMark[j] = 1, phi[j] = phi[j] / i * (i - 1);
}

int Phi(int n) {
    int ret = n;
    for(int i=2; i*i<=n; i++) if(n%i==0) {
            while (n%i==0) n/=i;
            ret -= ret/i;
        }
    return ret;
}

int bigMod(int a, int b, int M) {
    if(b==0) return 1%M;
    int x = bigMod(a, b/2, M);
    x = (x*x)%M;
    if(b%2==1) x = (x*a) % M;
    return x;
}

int egcd(int a, int b, int &x, int &y) {
    if(a==0) {
        x=0;y=1;
        return b;
    }
    int x1, y1;
    int d = egcd(b%a,a, x1, y1);
    x = y1 - (b/a) * x1, y = x1;
    return d;
}

void Ncr(int **ncr, int limncr) {
    for(int i=1; i<=limncr; i++)
        for(int j=0; j<=limncr; j++)
            if(j>i) ncr[i][j]=0;
            else if(j==i||j==0) ncr[i][j]=1;
            else ncr[i][j]=ncr[i-1][j-1]+ncr[i-1][j];
}

int num[100000], temp[100000];
void mergeSort(int lo, int hi) {
    if(lo==hi) return;
    int mid = (lo+hi)/2;
    mergeSort(lo, mid);
    mergeSort(mid+1, hi);
    for(int i=lo, j=mid+1, k=lo; k<=hi; k++) {
        if(i==mid+1) temp[k]=num[j++];
        else if(j==hi+1) temp[k]=num[i++];
        else if(num[i]<num[j]) temp[k]=num[i++];
        else temp[k]=num[j++];
    }
    for(int k=lo; k<=hi; k++) num[k]=temp[k];
}


int used[20], number[20], N=20;
void permutation(int at=1, int n=N) {
    if(at==n+1) {
        for(int i=1;i<=n; i++) printf("%d ", number[i]);
        printf("\n");
        return;
    }
    for(int i=1; i<=n; i++) if(!used[i]) {
        used[i]=1;
        number[at]=1;
        permutation(at+1, n);
        used[i]=0;
    }
}

//int number[20];
int n, k;
void combination(int at=1, int left=k) {
    if(left>n-at+1) return;
    if(at==n+1) {
        for(int i=1;i<=k; i++) printf("%d ", number[i]);
        printf("\n");
        return;
    }
    if(left) {
        number[k-left+1]=at;
        combination(at+1,left-1);
    }
    combination(at+1, left);
}

void combination2(int at=1, int last=0) {
    if(at==k+1) {
        for(int i=1; i<=k; i++) printf("%d ", number[i]);
        printf("\n");
        return;
    }
    for(int i=last+1; i<=n-k+at; i++) {
        number[at]=i;
        combination2(at+1, i);
    }
}

int queen[20];
int column[20], diagonal1[40], diagonal2[40];
void nqueen(int at=1, int n=8) {
    if(at==n+1) {
        printf("(row, column) = ");
        for(int i=1; i<=n; i++) printf("(%d, %d) ", i, queen[i]);
        printf("\n");
        return;
    }
    for(int i=1;i<=n; i++) {
        if(column[i]||diagonal1[i+at]||diagonal2[n+i-at]) continue;
        queen[at]=i;
        column[i]=diagonal1[i+at] = diagonal2[n+i-at]=1;
        nqueen(at+1, n);
        column[i]=diagonal1[i+at]=diagonal2[n+i-at]=0;
    }
}

int head[10000], data[100000], nxt[100000], id;
void insert(int x, int y) {
    data[id]=y;
    nxt[id]=head[x];
    head[x]=id;
    id++;
}
void erase(int x) {
    head[x]==nxt[head[x]];
}
int search(int x, int y) {
    for(int p=head[x]; p!=-1; p=nxt[p]) {
        if(data[p]==y) return 1;
    }
    return 0;
}

int p[100];
int Find(int x) {
    if(p[x]==x) return x;
    return p[x]= Find(p[x]);
}
void Union(int a, int b) {
    p[Find(b)] = b;
}

int on[10], toggle[10];
void build(int at, int L, int R) {
    on[at] = 0;
    if(L==R) {
        on[at]=0;
        return;
    }
    int mid = (L+R)/2;
    build(at*2, L, mid);
    build(at*2+1, mid+1, R);
    on[at]= on[at * 2] + on[at * 2 + 1];
}
void propagate(int at, int L, int R) {
    int mid = (L+R)/2;
    int left = at*2, right=at*2+1;
    toggle[at]=0;
    toggle[left] ^=1;
    toggle[right] ^=1;
    on[left] = (mid-L+1) - on[left];
    on[right]= (R-(mid+1)+1) - on[right];
}
void update(int at, int L, int R, int pos, int u) {
    if(L==R) {
        on[at]+=u;
        return;
    }
    if(pos<L || R<pos) return;
    if(toggle[at]) propagate(at, L, R);
    int mid = (L+R)/2;
    update(at*2, L, mid, pos, u);
    update(at*2+1, mid+1, R, pos, u);
    on[at]= on[at * 2] + on[at * 2 + 1];
}
int query(int at, int L, int R, int l, int r) {
    if(r<L|| R<l) return 0;
    if(l<=L&&R<=r) return on[at];
    if(toggle[at]) propagate(at, L, R);
    int mid = (L+R)/2;
    int x = query(at*2, L, mid, l, r);
    int y = query(at*2+1, mid+1, R, l, r);
    return x+y;
}

int tree[100], maxVal;
int read(int idx) {
    int sum=0;
    while (idx > 0) {
        sum += tree[idx];
        idx -= (idx&-idx);
    }
    return sum;
}
int update(int idx, int val) {
    while (idx <= maxVal) {
        tree[idx] += val;
        idx += (idx&-idx);
    }
}

typedef pair<int, int> PII;
vector<PII> v[100];
struct Node{
    int u, cost;
    Node(){}
    Node(int _u, int _cost) {
        u = _u;
        cost = _cost;
    }
};
bool operator<(Node A, Node B) {
    return A.cost > B.cost;
}
int cost[100], taken[100], s;
priority_queue<Node> PQ;
int prim() {
    for(int i=0; i<n; i++) cost[i]=1000000, taken[i]=0;
    cost[s]=0;
    PQ.push(Node(s, 0));
    int ans = 0;
    while (!PQ.empty()) {
        Node x = PQ.top();
        PQ.pop();
        if(taken[x.u]) continue;
        taken[x.u]=1;
        ans += x.cost;
        for(PII V: v[x.u]) {
            if(taken[V.first]) continue;
            if(V.second<cost[V.first]) {
                PQ.push(Node(V.first, V.second));
                cost[V.first] = V.second;
            }
        }
    }
    return ans;
}

struct Edge {
    int u, v, w;
};
bool operator<(Edge A, Edge B) {
    return A.w < B.w;
}
vector<Edge> E;
int kruskal() {
    sort(E.begin(),  E.end());
    int sz=E.size(), ans=0;
    for (int i = 0; i < sz; ++i) {
        if(Find(E[i].u)!= Find(E[i].v)) {
            p[p[E[i].u]] = p[E[i].v];
            ans += E[i].w;
        }
    }
    return ans;
}

int freq[100];
int huffman() {
    priority_queue<int, vector<int>, int> pq;
    for(int i=0; i<n; i++) pq.push(freq[i]);
    while (pq.size()!=1) {
        int a = pq.top();
        pq.pop();
        int b = pq.top();
        pq.pop();
        pq.push(a+b);
    }
    return pq.top();
}

vector<Edge> adj[100];
int dist[100];
void dijkstra(int s) {
    for(int i=0; i<n; i++) dist[i]=10000000;
    dist[s]=0;
    PQ.push(Node(s, 0));
    while (!PQ.empty()) {
        Node u = PQ.top();
        PQ.pop();
        if(u.cost!=dist[u.u]) continue;
        for(Edge e: adj[u.u]) {
            if (dist[e.v] > u.cost + e.w) {
                dist[e.v] = u.cost + e.w;
                PQ.push(Node(e.v, dist[e.v]));
            }
        }
    }
}
void bellmanFord(int s) {
    for(int i=0; i<n; i++) dist[i]=1000000;
    dist[s]=0;
    for(int i=0; i<n; i++) {
        for(Edge e: E) {
            if(dist[e.v]>dist[e.u]+e.w){
                dist[e.v]=dist[e.u]+e.w;
            }
        }
    }
}

int w[100][100];
void warshall() {
    for(int k=1; k<=n; k++) {
        for(int i=1; i <= n; i++) {
            for(int j=1; j<=n; j++) {
                if(w[i][j]>w[i][k]+w[k][j]) {
                    w[i][j]=w[i][k]+w[k][j];
                }
            }
        }
    }
}

int visited[100], matchR[100], matchL[100];
vector<int> V[100];
int bmpdfs(int y) {
    visited[y]=0;
    for(int i=0; i<V[y].size(); i++) {
        int z = V[y][i];
        if (matchR[z] == -1 || (!visited[matchR[z]] && bmpdfs(matchR[z]))) {
            matchL[y]=z;
            matchR[z]=y;
            return 1;
        }
    }
    return 0;
}

int pi[100], z[100];
char P[100], T[100];
void prefixFunction() {
    int now, len= strlen(P);
    pi[0]=now=-1;
    for(int i=1; i<len; i++) {
        while (now != -1 && P[now + 1] != P[i]) now=pi[now];
        if(P[now+1]==P[i]) pi[i]=++now;
        else pi[i]=now=-1;
    }
}
int kmp() {
    int now, n= strlen(T), m= strlen(P);
    for(int i=0; i<n; i++) {
        while (now!=-1&&P[now+1]!=T[i]) now=pi[now];
        if(P[now+1]==T[i]) ++now;
        else now=-1;
        if(now==m) return 1;
    }
    return 0;
}
void zfunction() {
    int left, right, len= strlen(P);
    z[0]=left=right=0;
    for(int i=1; i<len; i++) {
        if(i<=right) z[i]=min(z[i-left], z[right-i+1]);
        while (i+z[i]<len && P[i+z[i]==P[z[i]]]) z[i]++;
        if(i+z[i]-1>right) left=i, right=i+z[i]-1;
    }
}

char S[100];
int node[100][26], root, nnode, isWord[100];
void initialize() {
    root=0, nnode=0;
    for(int i=0; i<100; i++) node[root][i]=-1;
}
void insert() {
    scanf("%s", S);
    int len= strlen(S);
    int now=root;
    for(int i=0; i<len; i++) {
        if (node[now][S[i] - 'a'] == -1) {
            node[now][S[i]-'a']=++nnode;
            for(int j=0;j<26; j++) node[nnode][j]=-1;
        }
        now = node[now][S[i]-'a'];
    }
    isWord[now]=1;
}

int lcp[100], A[100], Rank[100];
void LCP() {
    int n= strlen(S);
    int now=0;
    for(int i=0; i<n; i++) Rank[A[i]]=i;
    for(int i=0; i<n; i++) {
        now = max(now-1, 0);
        if(Rank[i]==n-1){
            now = 0;
            continue;
        }
        int j=Rank[A[i]+1];
        while (i+now<n && j+now<n && S[i+now] == S[j+now]) now++;
        lcp[A[i]] = now;
    }
}
typedef vector<int> vi;

class UnionFind { // OOP style
private:
    vi p, rank; // remember: vi is vector<int>
public:
    UnionFind(int N) {
        rank.assign(N, 0);
        p.assign(N, 0);
        for (int i = 0; i < N; i++) p[i] = i;
    }

    int findSet(int i) { return (p[i] == i) ? i : (p[i] = findSet(p[i])); }

    bool isSameSet(int i, int j) { return findSet(i) == findSet(j); }

    void unionSet(int i, int j) {
        if (!isSameSet(i, j)) { // if from different set
            int x = findSet(i), y = findSet(j);
            if (rank[x] > rank[y]) p[y] = x; // rank keeps the tree short
            else {
                p[x] = y;
                if (rank[x] == rank[y]) rank[y]++;
            }
        }
    }
};

class SegmentTree { // the segment tree is stored like a heap array
private:
    vi st, A; // recall that vi is: typedef vector<int> vi;
    int n;

    int left(int p) { return p << 1; } // same as binary heap operations
    int right(int p) { return (p << 1) + 1; }

    void build(int p, int L, int R) { // O(n)
        if (L == R) // as L == R, either one is fine
            st[p] = L; // store the index
        else { // recursively compute the values
            build(left(p), L, (L + R) / 2);
            build(right(p), (L + R) / 2 + 1, R);
            int p1 = st[left(p)], p2 = st[right(p)];
            st[p] = (A[p1] <= A[p2]) ? p1 : p2;
        }
    }

    int rmq(int p, int L, int R, int i, int j) { // O(log n)
        if (i > R || j < L) return -1; // current segment outside query range
        if (L >= i && R <= j) return st[p]; // inside query range
        // compute the min position in the left and right part of the interval
        int p1 = rmq(left(p), L, (L + R) / 2, i, j);
        int p2 = rmq(right(p), (L + R) / 2 + 1, R, i, j);
        if (p1 == -1) return p2; // if we try to access segment outside query
        if (p2 == -1) return p1; // same as above
        return (A[p1] <= A[p2]) ? p1 : p2; // as in build routine
    }

public:
    SegmentTree(const vi &_A) {
        A = _A;
        n = (int) A.size(); // copy content for local usage
        st.assign(4 * n, 0); // create large enough vector of zeroes
        build(1, 0, n - 1); // recursive build
    }

    int rmq(int i, int j) { return rmq(1, 0, n - 1, i, j); } // overloading
};

void tain() {
    int arr[] = {18, 17, 13, 19, 15, 11, 20}; // the original array
    vi A(arr, arr + 7);
    SegmentTree st(A);
    printf("RMQ(1, 3) = %d\n", st.rmq(1, 3)); // answer = index 2
    printf("RMQ(4, 6) = %d\n", st.rmq(4, 6)); // answer = index 5
}

class FenwickTree {
private:
    vi ft; // recall that vi is: typedef vector<int> vi;
public:
    FenwickTree(int n) { ft.assign(n + 1, 0); } // init n + 1 zeroes

    int LSOne(int i) {
        return i&(-i);
    }

    int rsq(int b) { // returns RSQ(1, b)
        int sum = 0;
        for (; b; b -= LSOne(b)) sum += ft[b];
        return sum;
    } // note: LSOne(S) (S & (-S))
    int rsq(int a, int b) { // returns RSQ(a, b)
        return rsq(b) - (a == 1 ? 0 : rsq(a - 1));
    }

    // adjusts value of the k-th element by v (v can be +ve/inc or -ve/dec)
    void adjust(int k, int v) { // note: n = ft.size() - 1
        for (; k < (int) ft.size(); k += LSOne(k)) ft[k] += v;
    }
};

int ttain() {
    int f[] = {2, 4, 5, 5, 6, 6, 6, 7, 7, 8, 9}; // m = 11 scores
    FenwickTree ft(10); // declare a Fenwick Tree for range [1..10]
    // insert these scores manually one by one into an empty Fenwick Tree
    for (int i = 0; i < 11; i++) ft.adjust(f[i], 1); // this is O(k log n)
    printf("%d\n", ft.rsq(1, 1)); // 0 => ft[1] = 0
    printf("%d\n", ft.rsq(1, 2)); // 1 => ft[2] = 1
    printf("%d\n", ft.rsq(1, 6)); // 7 => ft[6] + ft[4] = 5 + 2 = 7
    printf("%d\n", ft.rsq(1, 10)); // 11 => ft[10] + ft[8] = 1 + 10 = 11
    printf("%d\n", ft.rsq(3, 6)); // 6 => rsq(1, 6) - rsq(1, 2) = 7 - 1
    ft.adjust(5, 2); // update demo
    printf("%d\n", ft.rsq(1, 10)); // now 13
} // return 0;

int dr[] = {1,1,0,-1,-1,-1, 0, 1}; // trick to explore an implicit 2D grid
int dc[] = {0,1,1, 1, 0,-1,-1,-1}; // S,SE,E,NE,N,NW,W,SW neighbors
int grid[100][100], R=100, C=100;
int floodfill(int r, int c, char c1, char c2) { // returns the size of CC
    if (r < 0 || r >= R || c < 0 || c >= C) return 0; // outside grid
    if (grid[r][c] != c1) return 0; // does not have color c1
    int ans = 1; // adds 1 to ans because vertex (r, c) has c1 as its color
    grid[r][c] = c2; // now recolors vertex (r, c) to c2 to avoid cycling!
    for (int d = 0; d < 8; d++)
        ans += floodfill(r + dr[d], c + dc[d], c1, c2);
    return ans; // the code is neat due to dr[] and dc[]
}

vi ts; // global vector to store the toposort in reverse order
vector<pair<int, int>> AdjList[100];
int dfs_num[100], VISITED = 1, UNVISITED = 0;

void dfs2(int u) { // different function name compared to the original dfs
    dfs_num[u] = VISITED;
    for (int j = 0; j < (int) AdjList[u].size(); j++) {
        pair<int, int> v = AdjList[u][j];
        if (dfs_num[v.first] == UNVISITED)
            dfs2(v.first);
    }
    ts.push_back(u);
}

typedef pair<int, int> ii;
int dfs_low[100], dfs_parent[100], dfsNumberCounter, dfsRoot, rootChildren, articulation_vertex[100];

void articulationPointAndBridge(int u) {
    dfs_low[u] = dfs_num[u] = dfsNumberCounter++; // dfs_low[u] <= dfs_num[u]
    for (int j = 0; j < (int) AdjList[u].size(); j++) {
        ii v = AdjList[u][j];
        if (dfs_num[v.first] == UNVISITED) { // a tree edge
            dfs_parent[v.first] = u;
            if (u == dfsRoot) rootChildren++; // special case if u is a root
            articulationPointAndBridge(v.first);
            if (dfs_low[v.first] >= dfs_num[u]) // for articulation point
                articulation_vertex[u] = true; // store this information first
            if (dfs_low[v.first] > dfs_num[u]) // for bridge
                printf(" Edge (%d, %d) is a bridge\n", u, v.first);
            dfs_low[u] = min(dfs_low[u], dfs_low[v.first]); // update dfs_low[u]
        } else if (v.first != dfs_parent[u]) // a back edge and not direct cycle
            dfs_low[u] = min(dfs_low[u], dfs_num[v.first]); // update dfs_low[u]
    }
}

void fain() {
    dfsNumberCounter = 0;
//    dfs_num.assign(V, UNVISITED); dfs_low.assign(V, 0);dfs_parent.assign(V, 0); articulation_vertex.assign(V, 0);
    printf("Bridges:\n");
    for (int i = 0; i < 100; i++)
        if (dfs_num[i] == UNVISITED) {
            dfsRoot = i;
            rootChildren = 0;
            articulationPointAndBridge(i);
            articulation_vertex[dfsRoot] = (rootChildren > 1);
        }
}

vi SS; // global variables
int numSCC = 0;

void tarjanSCC(int u) {
    dfs_low[u] = dfs_num[u] = dfsNumberCounter++; // dfs_low[u] <= dfs_num[u]
    SS.push_back(u); // stores u in a vector based on order of visitation
    visited[u] = 1;
    for (int j = 0; j < (int) AdjList[u].size(); j++) {
        ii v = AdjList[u][j];
        if (dfs_num[v.first] == UNVISITED)
            tarjanSCC(v.first);
        if (visited[v.first]) // condition for update
            dfs_low[u] = min(dfs_low[u], dfs_low[v.first]);
    }
    if (dfs_low[u] == dfs_num[u]) { // if this is a root (start) of an SCC
        printf("SCC %d:", ++numSCC); // this part is done after recursion
        while (1) {
            int v = SS.back();
            SS.pop_back();
            visited[v] = 0;
            printf(" %d", v);
            if (u == v) break;
        }
        printf("\n");
    }
}
void gain() {
//    dfs_num.assign(V, UNVISITED); dfs_low.assign(V, 0); visited.assign(V, 0);
    dfsNumberCounter = numSCC = 0;
    for (int i = 0; i < 100; i++)
        if (dfs_num[i] == UNVISITED)
            tarjanSCC(i);
}

void Kruskal() {
    // inside int main()
    vector<pair<int, ii> > EdgeList; // (weight, two vertices) of the edge
    int E = 0, u, v, w;
    for (int i = 0; i < E; i++) {
        scanf("%d %d %d", &u, &v, &w); // read the triple: (u, v, w)
        EdgeList.push_back(make_pair(w, ii(u, v)));
    } // (w, u, v)
    sort(EdgeList.begin(), EdgeList.end()); // sort by edge weight O(E log E)
    // note: pair object has built-in comparison function
    int mst_cost = 0;
    UnionFind UF(100); // all V are disjoint sets initially
    for (int i = 0; i < E; i++) { // for each edge, O(E)
        pair<int, ii> front = EdgeList[i];
        if (!UF.isSameSet(front.second.first, front.second.second)) { // check
            mst_cost += front.first; // add the weight of e to MST
            UF.unionSet(front.second.first, front.second.second); // link them
        }
    } // note: the runtime cost of UFDS is very light
    // note: the number of disjoint sets must eventually be 1 for a valid MST
    printf("MST cost = %d (Kruskal’s)\n", mst_cost);
}

int primes[100];
vi primeFactors(ll N) { // remember: vi is vector<int>, ll is long long
    vi factors;
    ll PF_idx = 0, PF = primes[PF_idx]; // primes has been populated by sieve
    while (PF * PF <= N) { // stop at sqrt(N); N can get smaller
        while (N % PF == 0) { N /= PF; factors.push_back(PF); } // remove PF
        PF = primes[++PF_idx]; // only consider primes!
    }
    if (N != 1) factors.push_back(N); // special case if N is a prime
    return factors; // if N does not fit in 32-bit integer and is a prime
}
void pain() {
    // inside int main(), assuming sieve(1000000) has been called before
    vi r = primeFactors(2147483647); // slowest, 2147483647 is a prime
    for (vi::iterator i = r.begin(); i != r.end(); i++) printf("> %d\n", *i);
    r = primeFactors(136117223861LL); // slow, 104729*1299709
    for (vi::iterator i = r.begin(); i != r.end(); i++) printf("# %d\n", *i);
    r = primeFactors(142391208960LL); // faster, 2^10*3^4*5*7^4*11*13
    for (vi::iterator i = r.begin(); i != r.end(); i++) printf("! %d\n", *i);
}

ll numPF(ll N) {
    ll PF_idx = 0, PF = primes[PF_idx], ans = 0;
    while (PF * PF <= N) {
        while (N % PF == 0) { N /= PF; ans++; }
        PF = primes[++PF_idx];
    }
    if (N != 1) ans++;
    return ans;
}

ll numDiv(ll N) {
    ll PF_idx = 0, PF = primes[PF_idx], ans = 1; // start from ans = 1
    while (PF * PF <= N) {
        ll power = 0; // count the power
        while (N % PF == 0) { N /= PF; power++; }
        ans *= (power + 1); // according to the formula
        PF = primes[++PF_idx];
    }
    if (N != 1) ans *= 2; // (last factor has pow = 1, we add 1 to it)
    return ans;
}

ll sumDiv(ll N) {
    ll PF_idx = 0, PF = primes[PF_idx], ans = 1; // start from ans = 1
    while (PF * PF <= N) {
        ll power = 0;
        while (N % PF == 0) { N /= PF; power++; }
        ans *= ((ll)pow((double)PF, power + 1.0) - 1) / (PF - 1);
        PF = primes[++PF_idx];
    }
    if (N != 1) ans *= ((ll)pow((double)N, 2.0) - 1) / (N - 1); // last
    return ans;
}

ll EulerPhi(ll N) {
    ll PF_idx = 0, PF = primes[PF_idx], ans = N; // start from ans = N
    while (PF * PF <= N) {
        if (N % PF == 0) ans -= ans / PF; // only count unique factor
        while (N % PF == 0) N /= PF;
        PF = primes[++PF_idx];
    }
    if (N != 1) ans -= ans / N; // last factor
    return ans;
}
int x, y, d;
void extendedEuclid(int a, int b) {
    if (b == 0) { x = 1; y = 0; d = a; return; } // base case
    extendedEuclid(b, a % b); // similar as the original gcd
    int x1 = y;
    int y1 = x - (a / b) * y;
    x = x1;
    y = y1;
}
ii floydCycleFinding(int x0) { // function int f(int x) is defined earlier
    // 1st part: finding k*mu, hare’s speed is 2x tortoise’s
    int tortoise = f(x0), hare = f(f(x0)); // f(x0) is the node next to x0
    while (tortoise != hare) { tortoise = f(tortoise); hare = f(f(hare)); }
    // 2nd part: finding mu, hare and tortoise move at the same speed
    int mu = 0; hare = x0;
    while (tortoise != hare) { tortoise = f(tortoise); hare = f(hare); mu++;}
    // 3rd part: finding lambda, hare moves, tortoise stays
    int lambda = 1; hare = f(tortoise);
    while (tortoise != hare) { hare = f(hare); lambda++; }
    return ii(mu, lambda);
}
ll numDiffPF(ll N) {
    ll PF_idx = 0, PF = primes[PF_idx], ans = 0;
    while (PF * PF <= N) {
        if (N % PF == 0) ans++; // count this pf only once
        while (N % PF == 0) N /= PF;
        PF = primes[++PF_idx];
    }
    if (N != 1) ans++;
    return ans;
}
ll sumPF(ll N) {
    ll PF_idx = 0, PF = primes[PF_idx], ans = 0;
    while (PF * PF <= N) {
        while (N % PF == 0) { N /= PF; ans += PF; }
        PF = primes[++PF_idx];
    }
    if (N != 1) ans += N;
    return ans;
}

#define MAX_N 100010
int b[MAX_N], m; // b = back table, n = length of T, m = length of P
void kmpPreprocess() { // call this before calling kmpSearch()
    int i = 0, j = -1;
    b[0] = -1; // starting values
    while (i < m) { // pre-process the pattern string P
        while (j >= 0 && P[i] != P[j]) j = b[j]; // different, reset j using b
        i++;
        j++; // if same, advance both pointers
        b[i] = j; // observe i = 8, 9, 10, 11, 12, 13 with j = 0, 1, 2, 3, 4, 5
    }
} // in the example of P = "SEVENTY SEVEN" above
void kmpSearch() { // this is similar as kmpPreprocess(), but on string T
    int i = 0, j = 0; // starting values
    while (i < n) { // search through string T
        while (j >= 0 && T[i] != P[j]) j = b[j]; // different, reset j using b
        i++;
        j++; // if same, advance both pointers
        if (j == m) { // a match found when j == m
            printf("P is found at index %d in T\n", i - j);
            j = b[j]; // prepare j for the next possible match
        }
    }
}
int RA[MAX_N], tempRA[MAX_N]; // rank array and temporary rank array
int SA[MAX_N], tempSA[MAX_N]; // suffix array and temporary suffix array
int c[MAX_N]; // for counting/radix sort
void countingSort(int k) { // O(n)
    int i, sum, maxi = max(300, n); // up to 255 ASCII chars or length of n
    memset(c, 0, sizeof c); // clear frequency table
    for (i = 0; i < n; i++) // count the frequency of each integer rank
        c[i + k < n ? RA[i + k] : 0]++;
    for (i = sum = 0; i < maxi; i++) {
        int t = c[i]; c[i] = sum; sum += t; }
    for (i = 0; i < n; i++) // shuffle the suffix array if necessary
        tempSA[c[SA[i]+k < n ? RA[SA[i]+k] : 0]++] = SA[i];
    for (i = 0; i < n; i++) // update the suffix array SA
        SA[i] = tempSA[i];
}
ii stringMatching() { // string matching in O(m log n)
    int lo = 0, hi = n-1, mid = lo; // valid matching = [0..n-1]
    while (lo < hi) { // find lower bound
        mid = (lo + hi) / 2; // this is round down
        int res = strncmp(T + SA[mid], P, m); // try to find P in suffix ’mid’
        if (res >= 0) hi = mid; // prune upper half (notice the >= sign)
        else lo = mid + 1; // prune lower half including mid
    } // observe ‘=’ in "res >= 0" above
    if (strncmp(T + SA[lo], P, m) != 0) return ii(-1, -1); // if not found
    ii ans; ans.first = lo;
    lo = 0; hi = n - 1; mid = lo;
    while (lo < hi) { // if lower bound is found, find upper bound
        mid = (lo + hi) / 2;
        int res = strncmp(T + SA[mid], P, m);
        if (res > 0) hi = mid; // prune upper half
        else lo = mid + 1; // prune lower half including mid
    } // (notice the selected branch when res == 0)
    if (strncmp(T + SA[hi], P, m) != 0) hi--; // special case
    ans.second = hi;
    return ans;
} // return lower/upperbound as first/second item of the pair, respectively
void constructSA() { // this version can go up to 100000 characters
    int i, k, r;
    for (i = 0; i < n; i++) RA[i] = T[i]; // initial rankings
    for (i = 0; i < n; i++) SA[i] = i; // initial SA: {0, 1, 2, ..., n-1}
    for (k = 1; k < n; k <<= 1) { // repeat sorting process log n times
        countingSort(k); // actually radix sort: sort based on the second item
        countingSort(0); // then (stable) sort based on the first item
        tempRA[SA[0]] = r = 0; // re-ranking; start from rank r = 0
        for (i = 1; i < n; i++) // compare adjacent suffixes
            tempRA[SA[i]] = // if same pair => same rank r; otherwise, increase r
                    (RA[SA[i]] == RA[SA[i-1]] && RA[SA[i]+k] == RA[SA[i-1]+k]) ? r : ++r;
        for (i = 0; i < n; i++) // update the rank array RA
            RA[i] = tempRA[i];
        if (RA[SA[n-1]] == n-1) break; // nice optimization trick
    } }

int smain() {
    n = (int) strlen(T); // input T as per normal, without the ‘$’
    T[n++] = '$'; // add terminating character
    constructSA();
    for (int i = 0; i < n; i++) printf("%2d\t%s\n", SA[i], T + SA[i]);
    while (m = (int) strlen(P)/*strlen(gets(P))*/, m) { // stop if P is an empty string
        ii pos = stringMatching();
        if (pos.first != -1 && pos.second != -1) {
            printf("%s found, SA [%d..%d] of %s\n", P, pos.first, pos.second, T);
            printf("They are:\n");
            for (int i = pos.first; i <= pos.second; i++)
                printf(" %s\n", T + SA[i]);
        } else printf("%s is not found in %s\n", P, T);
    }
} // return 0;

int PHI[100], PLCP[100], LCPP[100];
void computeLCP() {
    int i, L;
    PHI[SA[0]] = -1; // default value
    for (i = 1; i < n; i++) // compute PHI in O(n)
        PHI[SA[i]] = SA[i-1]; // remember which suffix is behind this suffix
        for (i = L = 0; i < n; i++) { // compute Permuted LCP in O(n)
            if (PHI[i] == -1) { PLCP[i] = 0; continue; } // special case
            while (T[i + L] == T[PHI[i] + L]) L++; // L increased max n times
            PLCP[i] = L;
            L = max(L-1, 0); // L decreased max n times
        }
        for (i = 0; i < n; i++) // compute LCP in O(n)
            LCPP[i] = PLCP[SA[i]]; // put the permuted LCP to the correct position
}

// struct point_i { int x, y; }; // basic raw form, minimalist mode
struct point_i {
    int x, y; // whenever possible, work with point_i
    point_i() { x = y = 0; } // default constructor
    point_i(int _x, int _y) : x(_x), y(_y) {}
}; // user-defined
struct point {
    double x, y; // only used if more precision is needed
    point() { x = y = 0.0; } // default constructor
    point(double _x, double _y) : x(_x), y(_y) {}
}; // user-defined
double distt(point p1, point p2) { // Euclidean distance
    // hypot(dx, dy) returns sqrt(dx * dx + dy * dy)
    return hypot(p1.x - p2.x, p1.y - p2.y);
} // return double
// rotate p by theta degrees CCW w.r.t origin (0, 0)
double DEG_to_RAD(double theta) {return theta * cos(-1.) / 180.0;}
point rotate(point p, double theta) {
    double rad = DEG_to_RAD(theta); // multiply theta with PI / 180.0
    return point(p.x * cos(rad) - p.y * sin(rad),
                 p.x * sin(rad) + p.y * cos(rad));
}
struct line { double a, b, c; };
double EPS = .00000000001;

void pointsToLine(point p1, point p2, line &l) {
    if (fabs(p1.x - p2.x) < EPS) { // vertical line is fine
        l.a = 1.0;
        l.b = 0.0;
        l.c = -p1.x; // default values
    } else {
        l.a = -(double) (p1.y - p2.y) / (p1.x - p2.x);
        l.b = 1.0; // IMPORTANT: we fix the value of b to 1.0
        l.c = -(double) (l.a * p1.x) - p1.y;
    }
}

bool areParallel(line l1, line l2) { // check coefficients a & b
    return (fabs(l1.a - l2.a) < EPS) && (fabs(l1.b - l2.b) < EPS);
}
bool areSame(line l1, line l2) { // also check coefficient c
    return areParallel(l1, l2) && (fabs(l1.c - l2.c) < EPS);
}

// returns true (+ intersection point) if two lines are intersect
bool areIntersect(line l1, line l2, point &p) {
    if (areParallel(l1, l2)) return false; // no intersection
    // solve system of 2 linear algebraic equations with 2 unknowns
    p.x = (l2.b * l1.c - l1.b * l2.c) / (l2.a * l1.b - l1.a * l2.b);
    // special case: test for vertical line to avoid division by zero
    if (fabs(l1.b) > EPS) p.y = -(l1.a * p.x + l1.c);
    else p.y = -(l2.a * p.x + l2.c);
    return true;
}

struct vec {
    double x, y; // name: ‘vec’ is different from STL vector
    vec(double _x, double _y) : x(_x), y(_y) {}
};
vec toVec(point a, point b) { // convert 2 points to vector a->b
    return vec(b.x - a.x, b.y - a.y);
}
vec scale(vec v, double s) { // nonnegative s = [<1 .. 1 .. >1]
    return vec(v.x * s, v.y * s);
} // shorter.same.longer
point translate(point p, vec v) { // translate p according to v
    return point(p.x + v.x, p.y + v.y);
}
double dot(vec a, vec b) { return (a.x * b.x + a.y * b.y); }
double norm_sq(vec v) { return v.x * v.x + v.y * v.y; }
// returns the distance from p to the line defined by
// two points a and b (a and b must be different)
// the closest point is stored in the 4th parameter (byref)
double distToLine(point p, point a, point b, point &c) {
    // formula: c = a + u * ab
    vec ap = toVec(a, p), ab = toVec(a, b);
    double u = dot(ap, ab) / norm_sq(ab);
    c = translate(a, scale(ab, u)); // translate a to c
    return distt(p, c);
} // Euclidean distance between p and c\

// returns the distance from p to the line segment ab defined by
// two points a and b (still OK if a == b)
// the closest point is stored in the 4th parameter (byref)
double distToLineSegment(point p, point a, point b, point &c) {
    vec ap = toVec(a, p), ab = toVec(a, b);
    double u = dot(ap, ab) / norm_sq(ab);
    if (u < 0.0) { c = point(a.x, a.y); // closer to a
    return distt(p, a); } // Euclidean distance between p and a
    if (u > 1.0) { c = point(b.x, b.y); // closer to b
    return distt(p, b); } // Euclidean distance between p and b
    return distToLine(p, a, b, c);
} // run distToLine as above

double angle(point a, point o, point b) { // returns angle aob in rad
    vec oa = toVec(o, a), ob = toVec(o, b);
    return acos(dot(oa, ob) / sqrt(norm_sq(oa) * norm_sq(ob)));
}
double cross(vec a, vec b) { return a.x * b.y - a.y * b.x; }
// note: to accept collinear points, we have to change the ‘> 0’
// returns true if point r is on the left side of line pq
bool ccw(point p, point q, point r) {
    return cross(toVec(p, q), toVec(p, r)) > 0; }
    // returns true if point r is on the same line as the line pq
    bool collinear(point p, point q, point r) {
    return fabs(cross(toVec(p, q), toVec(p, r))) < EPS;
}
int insideCircle(point_i p, point_i c, int r) { // all integer version
    int dx = p.x - c.x, dy = p.y - c.y;
    int Euc = dx * dx + dy * dy, rSq = r * r; // all integer
    return Euc < rSq ? 0 : Euc == rSq ? 1 : 2;
} //inside/border/outside
bool circle2PtsRad(point p1, point p2, double r, point &c) {
    double d2 = (p1.x - p2.x) * (p1.x - p2.x) +
            (p1.y - p2.y) * (p1.y - p2.y);
    double det = r * r / d2 - 0.25;
    if (det < 0.0) return false;
    double h = sqrt(det);
    c.x = (p1.x + p2.x) * 0.5 + (p1.y - p2.y) * h;
    c.y = (p1.y + p2.y) * 0.5 + (p2.x - p1.x) * h;
    return true;
} // to get the other center, reverse p1 and p2
double perimeter(double a, double b, double c) {
    return a+b+c;
}
double semiPerimeter(double a, double b, double c) {
    return perimeter(a, b, c) * 0.5;
}

double area(double a, double b, double c) {
    double s = semiPerimeter(a, b, c);
    return sqrt(s*(s-a)*(s-b)*(s-c));
}
double rInCircle(double ab, double bc, double ca) {
    return area(ab, bc, ca) / (0.5 * perimeter(ab, bc, ca)); }
    double rInCircle(point a, point b, point c) {
    return rInCircle(distt(a, b), distt(b, c), distt(c, a));
}
// assumption: the required points/lines functions have been written
// returns 1 if there is an inCircle center, returns 0 otherwise
// if this function returns 1, ctr will be the inCircle center
// and r is the same as rInCircle
int inCircle(point p1, point p2, point p3, point &ctr, double &r) {
    r = rInCircle(p1, p2, p3);
    if (fabs(r) < EPS) return 0; // no inCircle center
    line l1, l2; // compute these two angle bisectors
    double ratio = distt(p1, p2) / distt(p1, p3);
    point p = translate(p2, scale(toVec(p2, p3), ratio / (1 + ratio)));
    pointsToLine(p1, p, l1);
    ratio = distt(p2, p1) / distt(p2, p3);
    p = translate(p1, scale(toVec(p1, p3), ratio / (1 + ratio)));
    pointsToLine(p2, p, l2);
    areIntersect(l1, l2, ctr); // get their intersection point
    return 1;
}
double rCircumCircle(double ab, double bc, double ca) {
    return ab * bc * ca / (4.0 * area(ab, bc, ca)); }
    double rCircumCircle(point a, point b, point c) {
    return rCircumCircle(distt(a, b), distt(b, c), distt(c, a));
}
// returns the area, which is half the determinant
double area(const vector<point> &P) {
    double result = 0.0, x1, y1, x2, y2;
    for (int i = 0; i < (int)P.size()-1; i++) {
        x1 = P[i].x; x2 = P[i+1].x;
        y1 = P[i].y; y2 = P[i+1].y;
        result += (x1 * y2 - x2 * y1);
    }
    return fabs(result) / 2.0;
}
bool isConvex(const vector<point> &P) { // returns true if all three
    int sz = (int)P.size(); // consecutive vertices of P form the same turns
    if (sz <= 3) return false; // a point/sz=2 or a line/sz=3 is not convex
    bool isLeft = ccw(P[0], P[1], P[2]); // remember one result
    for (int i = 1; i < sz-1; i++) // then compare with the others
        if (ccw(P[i], P[i+1], P[(i+2) == sz ? 1 : i+2]) != isLeft)
            return false; // different sign -> this polygon is concave
            return true;
}
// returns true if point p is in either convex/concave polygon P
double PI= acos(-1.0);
bool inPolygon(point pt, const vector<point> &P) {
    if ((int)P.size() == 0) return false;
    double sum = 0; // assume the first vertex is equal to the last vertex
    for (int i = 0; i < (int)P.size()-1; i++) {
        if (ccw(pt, P[i], P[i+1]))
            sum += angle(P[i], pt, P[i+1]); // left turn/ccw
            else sum -= angle(P[i], pt, P[i+1]); } // right turn/cw
            return fabs(fabs(sum) - 2*PI) < EPS;
}
// line segment p-q intersect with line A-B.
point lineIntersectSeg(point p, point q, point A, point B) {
    double a = B.y - A.y;
    double b = A.x - B.x;
    double c = B.x * A.y - A.x * B.y;
    double u = fabs(a * p.x + b * p.y + c);
    double v = fabs(a * q.x + b * q.y + c);
    return point((p.x * v + q.x * u) / (u+v), (p.y * v + q.y * u) / (u+v)); }
    // cuts polygon Q along the line formed by point a -> point b
    // (note: the last point must be the same as the first point)
    vector<point> cutPolygon(point a, point b, const vector<point> &Q) {
    vector<point> P;
    for (int i = 0; i < (int)Q.size(); i++) {
        double left1 = cross(toVec(a, b), toVec(a, Q[i])), left2 = 0;
        if (i != (int)Q.size()-1) left2 = cross(toVec(a, b), toVec(a, Q[i+1]));
        if (left1 > -EPS) P.push_back(Q[i]); // Q[i] is on the left of ab
        if (left1 * left2 < -EPS) // edge (Q[i], Q[i+1]) crosses line ab
            P.push_back(lineIntersectSeg(Q[i], Q[i+1], a, b));
    }
    if (!P.empty() && !(P.back().x == P.front().x && P.back().y == P.front().y ))
        P.push_back(P.front()); // make P’s first point = P’s last point
        return P;
}
point pivot(0, 0);
bool angleCmp(point a, point b) { // angle-sorting function
    if (collinear(pivot, a, b)) // special case
        return distt(pivot, a) < distt(pivot, b); // check which one is closer
        double d1x = a.x - pivot.x, d1y = a.y - pivot.y;
        double d2x = b.x - pivot.x, d2y = b.y - pivot.y;
        return (atan2(d1y, d1x) - atan2(d2y, d2x)) < 0;
} // compare two angles
vector<point> CH(vector<point> P) { // the content of P may be reshuffled
    int i, j, n = (int)P.size();
    if (n <= 3) {
        if (!(P[0].x == P[n-1].x && P[0].y == P[n-1].y)) P.push_back(P[0]); // safeguard from corner case
        return P; } // special case, the CH is P itself
        // first, find P0 = point with lowest Y and if tie: rightmost X
        int P0 = 0;
    for (i = 1; i < n; i++)
        if (P[i].y < P[P0].y || (P[i].y == P[P0].y && P[i].x > P[P0].x))
            P0 = i;
        point temp = P[0]; P[0] = P[P0]; P[P0] = temp; // swap P[P0] with P[0]
        // second, sort points by angle w.r.t. pivot P0
        pivot = P[0]; // use this global variable as reference
        sort(++P.begin(), P.end(), angleCmp); // we do not sort P[0]
        // to be continued
        // continuation from the earlier part
        // third, the ccw tests
        vector<point> S;
        S.push_back(P[n-1]); S.push_back(P[0]); S.push_back(P[1]); // initial S
        i = 2; // then, we check the rest
        while (i < n) { // note: N must be >= 3 for this method to work
            j = (int)S.size()-1;
            if (ccw(S[j-1], S[j], P[i])) S.push_back(P[i++]); // left turn, accept
            else S.pop_back(); } // or pop the top of S until we have a left turn
            return S;
}
double INF=100000000;
struct line2 { double m, c; }; // another way to represent a line
int pointsToLine2(point p1, point p2, line2 &l) {
    if (p1.x == p2.x) { // special case: vertical line
        l.m = INF; // l contains m = INF and c = x_value
        l.c = p1.x; // to denote vertical line x = x_value
        return 0; // we need this return variable to differentiate result
    }
    else {
        l.m = (double)(p1.y - p2.y) / (p1.x - p2.x);
        l.c = p1.y - l.m * p1.x;
        return 1; // l contains m and c of the line equation y = mx + c
    }
}
// convert point and gradient/slope to line
void pointSlopeToLine(point p, double m, line &l) {
    l.a = -m; // always -m
    l.b = 1; // always 1
    l.c = -((l.a * p.x) + (l.b * p.y));
} // compute this
void closestPoint(line l, point p, point &ans) {
    line perpendicular; // perpendicular to l and pass through p
    if (fabs(l.b) < EPS) { // special case 1: vertical line
        ans.x = -(l.c); ans.y = p.y; return; }
    if (fabs(l.a) < EPS) { // special case 2: horizontal line
        ans.x = p.x; ans.y = -(l.c); return; }
    pointSlopeToLine(p, 1 / l.a, perpendicular); // normal line
    // intersect line l with this perpendicular line
    // the intersection point is the closest point
    areIntersect(l, perpendicular, ans);
}
// returns the reflection of point on a line
void reflectionPoint(line l, point p, point &ans) {
    point b;
    closestPoint(l, p, b); // similar to distToLine
    vec v = toVec(p, b); // create a vector
    ans = translate(translate(p, v), v);
} // translate p twice
#define MAX_N 100 // adjust this value as needed
struct AugmentedMatrix { double mat[MAX_N][MAX_N + 1]; };
struct ColumnVector { double vec[MAX_N]; };
ColumnVector GaussianElimination(int N, AugmentedMatrix Aug) { // O(N^3)
    // input: N, Augmented Matrix Aug, output: Column vector X, the answer
    int i, j, k, l; double t; ColumnVector X;
    for (j = 0; j < N - 1; j++) { // the forward elimination phase
        l = j;
        for (i = j + 1; i < N; i++) // which row has largest column value
            if (fabs(Aug.mat[i][j]) > fabs(Aug.mat[l][j]))
                l = i; // remember this row l
                // swap this pivot row, reason: to minimize floating point error
                for (k = j; k <= N; k++) // t is a temporary double variable
                    t = Aug.mat[j][k], Aug.mat[j][k] = Aug.mat[l][k], Aug.mat[l][k] = t;
                for (i = j + 1; i < N; i++) // the actual forward elimination phase
                    for (k = N; k >= j; k--)
                        Aug.mat[i][k] -= Aug.mat[j][k] * Aug.mat[i][j] / Aug.mat[j][j];
    }
    for (j = N - 1; j >= 0; j--) { // the back substitution phase
        for (t = 0.0, k = j + 1; k < N; k++) t += Aug.mat[j][k] * X.vec[k];
        X.vec[j] = (Aug.mat[j][N] - t) / Aug.mat[j][j]; // the answer is here
    }
    return X;
}
int DFS_WHITE=1;
void Kosaraju(int u, int pass) { // pass = 1 (original), 2 (transpose)
    dfs_num[u] = 1;
    vector<pair<int, int>> neighbor; // use different Adjacency List in the two passes
    if (pass == 1) neighbor = AdjList[u]; else neighbor = AdjList[u];
    for (int j = 0; j < (int)neighbor.size(); j++) {
        ii v = neighbor[j];
        if (dfs_num[v.first] == DFS_WHITE)
            Kosaraju(v.first, pass);
    }
    SS.push_back(u); // as in finding topological order in Section 4.2.5
}
void tttain() {
    // in int main()
    for (int i = 0; i < N; i++)
        if (dfs_num[i] == DFS_WHITE)
            Kosaraju(i, 1);
        numSCC = 0; // second pass: explore the SCCs based on first pass result
//        dfs_num.assign(N, DFS_WHITE);
        for (int i = N - 1; i >= 0; i--)
            if (dfs_num[S[i]] == DFS_WHITE) {
                numSCC++;
                Kosaraju(S[i], 2); // AdjListT -> the transpose of the original graph
            }
        printf("There are %d SCCs\n", numSCC);
}
#define MAX_N 10 // increase/decrease this value as needed
struct Matrix { int mat[MAX_N][MAX_N]; };
Matrix matMul(Matrix a, Matrix b, int p, int q, int r) { // O(pqr)
    Matrix c; int i, j, k;
    for (i = 0; i < p; i++)
        for (j = 0; j < r; j++)
            for (c.mat[i][j] = k = 0; k < q; k++)
                c.mat[i][j] += a.mat[i][k] + b.mat[k][j];
            return c;
}
int fastExp(int base, int p) { // O(log p)
    if (p == 0) return 1;
    else if (p == 1) return base; // See the Exercise below
    else { int res = fastExp(base, p / 2); res *= res;
        if (p % 2 == 1) res *= base;
        return res; }
}
Matrix matMul(Matrix a, Matrix b) { // O(n^3)
    Matrix ans; int i, j, k;
    for (i = 0; i < MAX_N; i++)
        for (j = 0; j < MAX_N; j++)
            for (ans.mat[i][j] = k = 0; k < MAX_N; k++) // if necessary, use
                ans.mat[i][j] += a.mat[i][k] * b.mat[k][j]; // modulo arithmetic
                return ans;
}
Matrix matPow(Matrix base, int p) { // O(n^3 log p)
    Matrix ans; int i, j;
    for (i = 0; i < MAX_N; i++) for (j = 0; j < MAX_N; j++)
        ans.mat[i][j] = (i == j); // prepare identity matrix
        while (p) { // iterative version of Divide & Conquer exponentiation
            if (p & 1) ans = matMul(ans, base); // if p is odd (last bit is on)
            base = matMul(base, base); // square the base
            p >>= 1; // divide p by 2
        }
        return ans;
}
void SlidingWindow(int A[], int n, int K) {
    // ii---or pair<int, int>---represents the pair (A[i], i)
    deque<ii> window; // we maintain ‘window’ to be sorted in ascending order
    for (int i = 0; i < n; i++) { // this is O(n)
        while (!window.empty() && window.back().first >= A[i])
            window.pop_back(); // this to keep ‘window’ always sorted
            window.push_back(ii(A[i], i));
            // use the second field to see if this is part of the current window
            while (window.front().second <= i - K) // lazy deletion
                window.pop_front();
            if (i + 1 >= K) // from the first window of length K onwards
                printf("%d\n", window.front().first); // the answer for this window
    }
}
#define MAX_N 1000 // adjust this value as needed
#define LOG_TWO_N 10 // 2^10 > 1000, adjust this value as needed
class RMQ { // Range Minimum Query
private:
    int _A[MAX_N], SpT[MAX_N][LOG_TWO_N];
public:
    RMQ(int n, int A[]) { // constructor as well as pre-processing routine
        for (int i = 0; i < n; i++) {
            _A[i] = A[i];
            SpT[i][0] = i; // RMQ of sub array starting at index i + length 2^0=1
        }
        // the two nested loops below have overall time complexity = O(n log n)
        for (int j = 1; (1<<j) <= n; j++) // for each j s.t. 2^j <= n, O(log n)
            for (int i = 0; i + (1<<j) - 1 < n; i++) // for each valid i, O(n)
                if (_A[SpT[i][j-1]] < _A[SpT[i+(1<<(j-1))][j-1]]) // RMQ
                    SpT[i][j] = SpT[i][j-1]; // start at index i of length 2^(j-1)
                    else // start at index i+2^(j-1) of length 2^(j-1)
                    SpT[i][j] = SpT[i+(1<<(j-1))][j-1];
    }
    int query(int i, int j) { // this query is O(1)
        int k = (int)floor(log((double)j-i+1) / log(2.0)); // 2^k <= (j-i+1)
        if (_A[SpT[i][k]] <= _A[SpT[j-(1<<k)+1][k]]) return SpT[i][k];
        else return SpT[j-(1<<k)+1][k];
    }
};
int main() {return 0;}
