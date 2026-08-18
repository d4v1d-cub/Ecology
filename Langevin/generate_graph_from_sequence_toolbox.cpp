#include "generate_graph_from_sequence_toolbox.h"

void save_matrix_to_file_sparse_ijaij(RealMatrix& graph, int N, FILE *fp_matrix){
    int i, j;
    for(i=0; i<N; i++){ 
      for(j=0; j<N; j++){
        if(graph(i, j)!=0)
            fprintf(fp_matrix, "%d\t%d\t%.17f\n", i, j, graph(i, j)); // 17 non è un numero completamente a caso, cfr Commenti/Printf.txt per maggiori dettagli
        }
    }
    return;
}

void save_matrix_to_file_sparse_ijaijaji(RealMatrix& graph, int N, FILE *fp_matrix){
    int i, j;
    // Stampa su file e su terminale
    for(i=0; i<N; i++){ 
      for(j=i+1; j<N; j++){
        if(graph(i, j)!=0)
            fprintf(fp_matrix, "%d\t%d\t%.17f\t%.17f\n", i, j, graph(i, j), graph(j, i)); // 17 non è un numero completamente a caso, cfr Commenti/Printf.txt per maggiori dettagli
      }
    }
    return;
}

void load_matrix_to_file_sparse_ijaij(RealMatrix& graph, int N, FILE *fp_matrix){
    int i, j;
    double aij;
    while(fscanf(fp_matrix, "%d\t%d\t%lf", &i, &j, &aij) > 0){
        graph(i, j) = aij;
    }
    return;
}

int build_crossing_index_table(int *crossindex, int *deg_seq, int fhs, int N){
    int i, k, kp1, kstar;
    bool kstar_notfound = MY_TRUE;
    if(fhs > N){
        printf("Not possible to build crossindex table of non-graphical sequence");
        exit(MY_TROUBLE);
    }
    crossindex[0] = N;
    kstar = fhs;
    k = 1;
    while(kstar_notfound){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i]){
            i = i-1;
        }
        crossindex[k] = i+1;
        if(crossindex[k] < kp1){
            kstar = k;
            kstar_notfound = MY_FALSE; // kstar should always exist
        }
        k = k+1;
    }
    // After finding kstar we keep building the crossing index table
    for(k=kstar+1; k<fhs; k++){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i]){
            i = i-1;
        }
        crossindex[k] = i+1;
    }
    crossindex[fhs] = 0; // In principle we have already used calloc but better be safe
    return kstar;
}

int build_crossing_index_table_remove_zeros(int *crossindex, int *deg_seq, int fhs, int N){
    int i, k, kp1, kstar, Nm1;
    bool kstar_notfound = MY_TRUE;
    Nm1 = N-1;
    while(deg_seq[Nm1]==0){
        N = Nm1;
        Nm1--;
    } // Here N is the number of non-zero degrees
    if(fhs > N){
        printf("Not possible to build crossindex table of non-graphical sequence");
        exit(MY_TROUBLE);
    }
    crossindex[0] = N;
    kstar = fhs;
    k = 1;
    while(kstar_notfound){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i]){
            i = i-1;
        }
        crossindex[k] = i+1;
        if(crossindex[k] < kp1){
            kstar = k;
            kstar_notfound = MY_FALSE; // kstar should always exist
        }
        k = k+1;
    }
    // After finding kstar we keep building the crossing index table
    for(k=kstar+1; k<fhs; k++){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i]){
            i = i-1;
        }
        crossindex[k] = i+1;
    }
    crossindex[fhs] = 0; // In principle we have already used calloc but better be safe
    return kstar;
}

int build_crossing_index_table_up_kstar(int *crossindex, int *deg_seq, int fhs, int N){
    int i, k, kp1, kstar;
    bool kstar_notfound = MY_TRUE;
    if(fhs > N){
        printf("Not possible to build crossindex table of non-graphical sequence");
        exit(MY_TROUBLE);
    }
    crossindex[0] = N;
    kstar = fhs;
    for(k=1; (k<N && kstar_notfound); k++){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i] && i > 0){
            i = i-1;
        }
        crossindex[k] = i+1;
        if(crossindex[k] < kp1){
            kstar = k;
            kstar_notfound = MY_FALSE; // kstar should always exist
            //k = fhs; // for clarity I prefer to use a bool instead of modifying k
        }
    }
    // Error handling
    if(kstar_notfound){
        fprintf(stderr, "ERROR: no kstar found before k reached N =%u\n", N);
        exit(MY_TROUBLE);
    }
    // In this case the behaviour of crossindex after kstar is uncontrolled, it is appropriate to use realloc at least
    return kstar;
}

void build_crossing_index_table_no_kstar(int *crossindex, int *deg_seq, int fhs, int N){
    int i, k, kp1;
    if(fhs > N){
        printf("Not possible to build crossindex table of non-graphical sequence");
        exit(MY_TROUBLE);
    }
    crossindex[0] = N;
    // Let's build the crossing index table
    for(k=1; k<fhs; k++){
        kp1 = k+1;
        i = crossindex[k-1]-1;
        while(kp1 > deg_seq[i]){
            i = i-1;
        }
        crossindex[k] = i+1;
    }
    crossindex[fhs] = 0; // In principle we have already used calloc but better be safe
    return;
}

bool erdos_gallai_test(int *deg_seq, int N, int *crossindex, int kstar){
    int k;
    double L, R;
    bool graphical = MY_TRUE;
    if((sum_array(deg_seq, N) % 2) == 1){ // The degrees sum is odd
        graphical = MY_FALSE;
    }
    L = deg_seq[0];
    R = N-1;
    if(R < L){
        graphical = MY_FALSE;
    }
    k = 1; //  The first step k=0 is in the initialisation of R and L
    while((k < kstar) && (graphical)){ // graphical==MY_TRUE is redundant if MY_TRUE is 1 (True)
        R += crossindex[k] - 1;
        L += deg_seq[k];
        if(R < L){
            graphical = MY_FALSE;
        }
        k++;
    }
    return graphical;
}

void build_dprime(int *rdsp, int *cip, int *ksp, int *fnpc, int *rds, int hs, int fhs, int *crossindex, int kstar){
    int kbar = rds[hs-1]; // One-to-Smallest degree in the Leftmost Adj Set 
    int nkbnc = crossindex[kbar-1] - crossindex[hs] - hs; // Nodes with degree kbar not connected to the hub = (crossindex[kbar-1] - 1) - (hubsize-1) = #(number of nodes within kbar except the hub) - #(new connections
    int i, start_new_kbar_pals, ksp1;
    fill_array(fnpc, 0, fhs+1);
    fnpc[1] = 1; // The hub itself
    // Fill reduced degree sequence prime
    if(kbar != hs)
        start_new_kbar_pals = crossindex[kbar]-1; 
    else
        start_new_kbar_pals = crossindex[hs];
    for(i=crossindex[hs]; i<start_new_kbar_pals; i++){
        rdsp[i] = rds[i+1] - 1;
        fnpc[rdsp[i]] += 1;
    }
    for(i=start_new_kbar_pals; i<start_new_kbar_pals+nkbnc; i++){
        rdsp[i] = kbar;
    }
    for(i=start_new_kbar_pals+nkbnc; i<crossindex[kbar-1]-1; i++){
        rdsp[i] = kbar - 1;
    }
    fnpc[kbar-1] += crossindex[hs] + hs - 1 - start_new_kbar_pals;
    for(i=crossindex[kbar-1]-1; i<crossindex[1]-1; i++){
        rdsp[i] = rds[i+1];
    }
    if(kbar > 1)
        for(i=crossindex[1]-1; i<crossindex[0]; i++)
            rdsp[i] = 1;
    else{
        //cip[0] = crossindex[0] - fnpc[0];
        rdsp[cip[0]-1] = 1;
        rdsp[crossindex[0]-1] = 0;
    }
    for(i=1; i<kbar-1; i++)
        cip[i] = crossindex[i] - 1;
    for(i=kbar; i<hs-1; i++)
        cip[i] = crossindex[i+1] - 1;
    for(i=hs-1; i<hs+1; i++)
        cip[i] = crossindex[hs];
    cip[kbar-1] = start_new_kbar_pals + nkbnc;
    cip[0] = crossindex[0] - fnpc[0];
    for(i=kbar+1; i<hs+1; i++){
        if(crossindex[i]==crossindex[hs])
            cip[i-1] = cip[hs];
    }
    // Let's find kstar prime (ksp)
    *ksp = kstar;
    ksp1 = *ksp-1;
    while(cip[ksp1]-cip[hs]<=ksp1){
        *ksp = ksp1;
        ksp1 -= 1;
    }
    return;
}

void update_rds_connection(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *fingerprint, int q, int hs){
    int dq = rds[q];
    int dqm1 = dq-1;
    int ksm1, cdqm1;
    // Reduce hub degree
    rds[crossindex[hs]] -= 1; // rds[crossindex[hs]] is the hub
    // Reduce chosen node degree
    crossindex[dqm1] -= 1; // We shift crossindex of degree dq-1
    cdqm1 = crossindex[dqm1];
    rds[cdqm1]--; // We reduce the last dq making it the first dq-1
    anc[dq]--;
    fnc[dqm1]++;
    swap((fingerprint+q), (fingerprint+cdqm1)); // We swap the fingerprints
    // Let's update kstar
    ksm1 = *kstar - 1;
    while(crossindex[ksm1]-crossindex[hs]<=ksm1){
        *kstar = ksm1;
        ksm1--;
    }
    return;
}

void update_rdsp_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int srd, int dqm1, int hs){
    if(dqm1 < srd+1){ // q is not within the hs-1 connected nodes, excluding forbidden nodes
        // Remove last prime-connection
        rdsp[cip[srd]]++;
        cip[srd]++;
        fnpc[srd]--;
        // Create new connection
        cip[dqm1]--;
        rdsp[cip[dqm1]]--;
        fnpc[dqm1]++;
        // Let's update kstar prime (ksp): in this case cip is both incremented and reduced so we need to check both directions
        int ksp1 = *ksp - 1; // I don't like the aesthetic but it is pointless to define the variable outside the if statement
        while(cip[ksp1]-cip[hs]<=ksp1){ // This implies cip[ksp-1]-cip[hs] > ksp-1
            *ksp = ksp1;
            ksp1--;
        }
        while(cip[*ksp]-cip[hs]>*ksp){ // This implies cip[ksp]-cip[hs] <= ksp ---> This together with the previous one means that ksp is the first one
            *ksp = *ksp + 1;
        }
    }
    return;
}

void update_rdsp_last_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int dqm1, int hs){
    int ksp1;
    // Create new connection
    cip[dqm1]--;
    rdsp[cip[dqm1]]--;
    fnpc[dqm1]++;
    // Let's update kstar prime (ksp)
    ksp1 = *ksp-1;
    while(cip[ksp1]-cip[hs]<=ksp1){ // This implies cip[ksp-1]-cip[hs] > ksp-1
        *ksp = ksp1;
        ksp1--;
    }
    return;
}

void update_rds_newhub(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *hs, int fhs){
    crossindex[*hs]++; // We shift crossindex of 1 to leave the exausted hub 
    *hs = rds[crossindex[*hs]]; // New hub size
    if(*hs > 0){
        // Let's update kstar
        int i, ksm1 = *kstar-1; // I don't like the aesthetic but it is pointless to define the variable outside the if statement
        while(crossindex[ksm1]-crossindex[*hs]<=ksm1){ // If we do not need kstar for rds after the EG test the update can be commented
            *kstar = ksm1;
            ksm1--;
        }
        for(i=1; i<*hs+1; i++){
            anc[i] = anc[i] + fnc[i];
        }
        fill_array(fnc, 0, fhs+1);
        anc[*hs]--;
    }
    return;
}

void update_rdsp_newhub(int *rdsp, int *cip, int *ksp, int *fnpc, int hs, int fhs){
    int i, kbar, nkbc, nkbnc, ksp1;
    // It is important to update kbar before removing the hub
    kbar = rdsp[cip[hs]+hs-1]; // One-to-Smallest degree in the Leftmost Adj Set
    rdsp[cip[hs]] = 0;
    cip[hs]++; // We shift cip of 1 to leave the exausted hub
    fill_array(fnpc, 0, fhs+1);
    fnpc[1] = 1; // The hub itself
    // Let's compute nkb
    nkbc = hs - 1 + cip[hs] - cip[kbar]; // Nodes with degree kbar connected to the hub = (hub_size-1) - (cip[kbar]-cip[hs]) = #(new connections) - #(nodes before kbar) # Notice that the hub is already at the end of the sequence
    // We should call them nodes kb that has remained kbar
    nkbnc = cip[kbar-1] - cip[hs] - hs + 1; // Nodes with degree kbar not connected to the hub = (cip[kbar-1]-cip[hs]) - (hubsize-1) = #(number of nodes within kbar except the hub) - #(new connections) # Notice that the hub is already at the end of the sequence
    for(i=cip[hs]; i<cip[kbar]; i++){
        rdsp[i]--;
        fnpc[rdsp[i]]++;
    }
    for(i=cip[kbar]+nkbnc; i<cip[kbar-1]; i++){ // It would be equivalent from i=cip[kbar-1]-nkbc to i=cip[kbar-1])
        rdsp[i] = kbar - 1;
    }
    // Update crossindex prime
    cip[kbar-1] = cip[kbar] + nkbnc; // It is important to use cip[kbar] before the update
    fnpc[kbar-1] += nkbc;
    for(i=kbar; i<hs; i++){
        cip[i] = cip[i+1];
    }
    for(i=kbar+1; i<hs; i++){
        cip[i] = cip[i-1] - fnpc[i];
    }
    // Let's update kstar prime (ksp)
    ksp1 = *ksp-1;
    while(cip[ksp1]-cip[hs]<=ksp1){
        *ksp = ksp1;
        ksp1--;
    }
    return;
}

void fail_degree_k(int L, int R, int *fdk, int *k, int ridx, int *rdsp, int *cip, int ciphs, int ksp, int *fnpc){ // In principle it would be enough to give R-L to the function 
    if(L < R-1) // Case 3
        *fdk = 0;
    else if(L == R-1){ // Case 2
        *fdk = rdsp[my_max(ciphs+*k+1, cip[*k+1])]; // First node whose index is greater than k and whose degree is smaller than k+2
        while((cip[*fdk-1] - cip[*fdk] - fnpc[*fdk]) == 0) // Check that there are any non-forbidden nodes with degree fdk
            *fdk = *fdk - 1;
    }
    else{ // Case 1 (it would be L==R and since L<=R that's the only case left)
        *fdk = rdsp[ciphs+*k+1]; // First node whose index is greater than k
        while((cip[*fdk-1] - cip[*fdk] - fnpc[*fdk]) == 0) // Check that there are any non-forbidden nodes with degree fdk
            *fdk = *fdk - 1;
        *k = ridx; // After Case (1) we don't have to check further values of k
    }
    return;
}

int find_maximum_fail_degree(int *rdsp, int *cip, int ciphs, int ksp, int *fnpc){
    int mfd, fdk, L, R, ridx, k;
    if(rdsp[ciphs+ksp] == ksp) // ridx = r index, to distinguish from R
        ridx = ksp+1;
    else
        ridx = ksp;
    L = rdsp[ciphs];
    R = cip[0]-ciphs - 1;
    k = 0;
    fail_degree_k(L, R, &mfd, &k, ridx, rdsp, cip, ciphs, ksp, fnpc); // No need to find max at this stage
    k = 1; // The first step k=0 is in the initialisation
    while(k < ksp){
        L += rdsp[ciphs+k];
        R += cip[k]-ciphs - 1;
        fail_degree_k(L, R, &fdk, &k, ridx, rdsp, cip, ciphs, ksp, fnpc);
        mfd = my_max(mfd, fdk);
        k++;
    }
    // After the previous loop either k=ksp if no (1) and therefore we need to keep checking (one or two times depending on ridx) or k=ridx+1 if (1) and therefore we are done
    while(k < ridx+1){ 
        L += rdsp[ciphs+k];
        R += 2*k - rdsp[ciphs+k];
        fail_degree_k(L, R, &fdk, &k, ridx, rdsp, cip, ciphs, ksp, fnpc);
        mfd = my_max(mfd, fdk);
        k++;
    }
    return mfd;
}

void build_allowed_nodes_counter(int *anc, int *crossindex, int *fnc, int hs){
    int i;
    for(i=1; i<hs; i++)
        anc[i] = crossindex[i-1] - crossindex[i] - fnc[i];
    anc[hs] = crossindex[hs-1] - crossindex[hs] - 1; // The -1 is the hub while we don't have - fnc[hs] since by construction we don't have any forbidden node of degree hs (a part from the hub)
    return;
}

void build_allowed_nodes_counter_at_beginning(int *anc, int *crossindex, int hs){
    int i;
    for(i=1; i<hs; i++)
        anc[i] = crossindex[i-1] - crossindex[i];
    anc[hs] = crossindex[hs-1] - crossindex[hs] - 1; // The -1 is the hub
    return;
}

void extract_allowed_node(int *eni, int *tnan, int *crossindex, int hs, int *anc, int *fnc, int mfd){
    int rfn, si, ps, q, i;
    si = my_max(mfd+1,1); // Starting index - &fnc[si] is equivalent to fnc+si
    rfn = sum_array(&fnc[si], hs+1-si); // We don't want to substract the nodes with zero degree, that's why the max # It would be enough to sum up to hs-1 since there are no forbidden nodes of degree hs, by construction
    *tnan = crossindex[mfd] - crossindex[hs] -  1 - rfn; // Total number of allowed nodes (the -1 is from the hub) # This is the size of the allowed set needed for the sample weights
    q = RandIntegers(0, *tnan-1); // We can also use q = rng.integers(1:tnan+1), while (ps < q) and eni = crossindex[hs] + q + fnc[i:hs+1].sum() (without the +1 from the hub)
    // Start from large degree
    ps = anc[hs];
    i = hs;
    while(ps <= q){ // ps < q+1
        i--;
        ps += anc[i]; // Extracted degree is i
    }
    *eni = crossindex[hs] + 1 + q + sum_array(&fnc[i], hs+1-i); // Extracted Node Index: the +1 is the hub // It would be enough to sum up to hs-1 since there are no forbidden nodes of degree hs, by construction
    return;
}

int find_smallest_reduced_degree(int *fnc, int *fnpc){
    int srd;
    if(fnc[0] != fnpc[0])
        srd = 0;
    else if(fnc[1] != (fnpc[1]-1)) // -1 because of the hub
        srd = 1;
    else{
        srd = 2;
        while(fnc[srd]==fnpc[srd])
            srd += 1;
    }
    return srd;
}

void graph_from_degree_sequence(RealMatrix& graph, int N, int *rds, int *rdsp, int *fingerprint, int *crossindex, int kstar, void (*weights_generator)(double*, double*, double, double, double), double p1, double p2, double epsilon){
    int *cip, *anc, *fnc, *fnpc;
    int fhs, hs, ksp, q, kij, dqm1, srd, hs_updated, ps, sdlas, mfd, i;
    double a_ij, a_ji; 

    //double logweight;
    fhs = rds[0];
    // Clean and reset everything
    graph.setZero();
    fill_array(rdsp, 0, N);
    for(i=0; i<N; i++){
        fingerprint[i] = i; // Fill in indexes
    }
    //logweight = 0;
    cip = my_int_calloc(fhs+1);
    anc = my_int_calloc(fhs+1); // # Allowed Nodes Counter: anc[i] = allowed nodes of degree i
    fnc = my_int_calloc(fhs+1); // Forbidden nodes counter (it includes forbidden zeros so that fnc[n] = number forbidden nodes of degree n)
    fnpc = my_int_calloc(fhs+1);
    // Let's start
    hs = fhs;
    build_allowed_nodes_counter_at_beginning(anc, crossindex, fhs);
    build_dprime(rdsp, cip, &ksp, fnpc, rds, hs, fhs, crossindex, kstar);
    while(hs>1){
        kij = crossindex[0] - crossindex[hs] - 1; // The first connection of an hub is always possible (except with itself, thus -1)
        q = crossindex[hs] + 1 + RandIntegers(0, kij-1); // The +1 is the hub
        //weight = weight / hs * kij;
        //logweight += log(kij) - log(hs);
        //weights_generator(&graph(fingerprint[crossindex[hs]], fingerprint[q]), &graph(fingerprint[q], fingerprint[crossindex[hs]]), p1, p2, epsilon);
        weights_generator(&a_ij, &a_ji, p1, p2, epsilon);
        graph(fingerprint[crossindex[hs]], fingerprint[q]) = (Real) a_ij;
        graph(fingerprint[q], fingerprint[crossindex[hs]]) = (Real) a_ji;
        dqm1 = rds[q]-1;
        srd = find_smallest_reduced_degree(fnc, fnpc); // Find Smallest Reduced Degree
        update_rds_connection(rds, crossindex, &kstar, anc, fnc, fingerprint, q, hs);
        update_rdsp_connection(rdsp, cip, &ksp, fnpc, srd, dqm1, hs);
        for(i=1; i<hs; i++){
            hs_updated = hs-i; // Updated Hub Size
            ps = anc[hs];
            sdlas = hs;
            while(ps < hs_updated){
                sdlas--;
                ps += anc[sdlas]; // Extracted degree is sdlas (Smallest degree in the Leftmost Adj Set)
            }
            mfd = 0;
            if(sdlas != rds[crossindex[0]-1])  // if sdlas==rds[-1] then mfd=0 
                mfd = find_maximum_fail_degree(rdsp, cip, cip[hs], ksp, fnpc);
            extract_allowed_node(&q, &kij, crossindex, hs, anc, fnc, mfd);
            dqm1 = rds[q]-1;
            //weight = weight / hs_updated * kij;
            //logweight += log(kij) - log(hs);
            //weights_generator(&graph(fingerprint[crossindex[hs]], fingerprint[q]), &graph(fingerprint[q], fingerprint[crossindex[hs]]), p1, p2, epsilon);
            weights_generator(&a_ij, &a_ji, p1, p2, epsilon);
            graph(fingerprint[crossindex[hs]], fingerprint[q]) = (Real) a_ij;
            graph(fingerprint[q], fingerprint[crossindex[hs]]) = (Real) a_ji;
            if(i<(hs-1)){ // "Bulk" of connections
                srd = find_smallest_reduced_degree(fnc, fnpc); // Find new Smallest Reduced Degree
                update_rdsp_connection(rdsp, cip, &ksp, fnpc, srd, dqm1, hs);
            }
            else // Last connection
                update_rdsp_last_connection(rdsp, cip, &ksp, fnpc, dqm1, hs);
            // We update rds after rdsp in order to easily compute srd
            update_rds_connection(rds, crossindex, &kstar, anc, fnc, fingerprint, q, hs);
        }
        // New Hub: we update hs
        update_rds_newhub(rds, crossindex, &kstar, anc, fnc, &hs, fhs);
        if(hs > 1)
            update_rdsp_newhub(rdsp, cip, &ksp, fnpc, hs, fhs);
    }
    // When there are only ones left there is no need to compute prime quantities
    while(hs != 0){
        kij = crossindex[0] - crossindex[1] - 1; // The first connection of an hub is always possible (except with itself, thus -1)
        q = crossindex[hs] + 1 + RandIntegers(0, kij-1); // The +1 is the hub
        //weight = weight * kij;
        //logweight += log(kij);
        //weights_generator(&graph(fingerprint[crossindex[hs]], fingerprint[q]), &graph(fingerprint[q], fingerprint[crossindex[hs]]), p1, p2, epsilon);
        weights_generator(&a_ij, &a_ji, p1, p2, epsilon);
        graph(fingerprint[crossindex[hs]], fingerprint[q]) = (Real) a_ij;
        graph(fingerprint[q], fingerprint[crossindex[hs]]) = (Real) a_ji;
        update_rds_connection(rds, crossindex, &kstar, anc, fnc, fingerprint, q, hs);
        // We update hs
        update_rds_newhub(rds, crossindex, &kstar, anc, fnc, &hs, fhs);
    }
    return;
}