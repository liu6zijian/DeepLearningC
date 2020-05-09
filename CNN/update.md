			cnnbp(cnn,outputData->LabelPtr[n].LabelData); // 后向传播，这里主要计算各神经元的误差梯度
			float*** temp_y;
			MPI_Reduce(cnn->C1->y,temp_y,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_d;
			MPI_Reduce(cnn->C1->d,temp_d,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_v;
			MPI_Reduce(cnn->C1->v,temp_v,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			if (myid==0){
				for(int i=0;i<cnn->C1->outChannels;i++){
					for(int r=0;r<cnn->S2->inputWidth;r++){
						for(int c=0;c<cnn->S2->inputHeight;c++){
							cnn->C1->y[i][r][c]= temp_y[i][r][c] / numprocs;
							cnn->C1->d[i][r][c]= temp_d[i][r][c] / numprocs;
							cnn->C1->v[i][r][c]= temp_v[i][r][c] / numprocs;
						}
					}
				}
			}
			free(temp_y);free(temp_d);free(temp_v);

			float*** temp_y;
			MPI_Reduce(cnn->C3->y,temp_y,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_d;
			MPI_Reduce(cnn->C3->d,temp_d,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_v;
			MPI_Reduce(cnn->C3->v,temp_v,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			if (myid==0){
				for(int i=0;i<cnn->C3->outChannels;i++){
					for(int r=0;r<cnn->S4->inputWidth;r++){
						for(int c=0;c<cnn->S4->inputHeight;c++){
							cnn->C3->y[i][r][c]= temp_y[i][r][c] / numprocs;
							cnn->C3->d[i][r][c]= temp_d[i][r][c] / numprocs;
							cnn->C3->v[i][r][c]= temp_v[i][r][c] / numprocs;
						}
					}	
				}
			}
			free(temp_y);free(temp_d);free(temp_v);

			float* temp_y;
			MPI_Reduce(cnn->O5->y,temp_y,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float* temp_d;
			MPI_Reduce(cnn->O5->d,temp_d,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float* temp_v;
			MPI_Reduce(cnn->O5->v,temp_v,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			if (myid==0){
				for(int i=0;i<cnn->O5->outputNum;i++){	
					cnn->C3->y[i]= temp_y[i] / numprocs;
					cnn->C3->d[i]= temp_d[i] / numprocs;
					cnn->C3->v[i]= temp_v[i] / numprocs;
				}
			}
			free(temp_y);free(temp_d);free(temp_v);

			float*** temp_y;
			MPI_Reduce(cnn->S2->y,temp_y,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_d;
			MPI_Reduce(cnn->S2->d,temp_d,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_v;
			MPI_Reduce(cnn->S2->v,temp_v,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			if (myid==0){
				for(int i=0;i<cnn->S2->outChannels;i++){
					for(int r=0;r<cnn->(S2->inputWidth/S2->mapSize);r++){
						for(int c=0;c<cnn->(S2->inputHeight/S2->mapSize);c++){
							cnn->S2->y[i][r][c]= temp_y[i][r][c] / numprocs;
							cnn->S2->d[i][r][c]= temp_d[i][r][c] / numprocs;
							cnn->S2->v[i][r][c]= temp_v[i][r][c] / numprocs;
						}
					}	
				}
			}
			free(temp_y);free(temp_d);free(temp_v);

			float*** temp_y;
			MPI_Reduce(cnn->S4->y,temp_y,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_d;
			MPI_Reduce(cnn->S4->d,temp_d,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			float*** temp_v;
			MPI_Reduce(cnn->S4->v,temp_v,MPI_FLOAT,MPI_SUM, 0,MPI_COMM_WORLD);
			if (myid==0){
				for(int i=0;i<cnn->S4->outChannels;i++){
					for(int r=0;r<cnn->(S4->inputWidth/S2->mapSize);r++){
						for(int c=0;c<cnn->(S4->inputHeight/S2->mapSize);c++){
							cnn->S4->y[i][r][c]= temp_y[i][r][c] / numprocs;
							cnn->S4->d[i][r][c]= temp_d[i][r][c] / numprocs;
							cnn->S4->v[i][r][c]= temp_v[i][r][c] / numprocs;
						}
					}	
				}
			}
			free(temp_y);free(temp_d);free(temp_v);
