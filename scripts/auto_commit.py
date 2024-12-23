import os
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import time
import traceback

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/arthur/auto_commit.log'),
        logging.StreamHandler()
    ]
)

class GitAutoCommit:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.max_file_size = 15 * 1024 * 1024  # 15 MB in bytes
        self.excluded_paths = ['Experiments']

    def should_exclude_file(self, file_path: str) -> bool:
        """Проверяет, должен ли файл быть исключен из коммита."""
        path = Path(file_path)
        return any(excluded in path.parts for excluded in self.excluded_paths)

    def run_git_command(self, command: list) -> tuple[int, str, str]:
        """Выполняет git команду и возвращает код возврата, stdout и stderr."""
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.repo_path)
        )
        stdout, stderr = process.communicate()
        return (
            process.returncode,
            stdout.decode('utf-8', errors='ignore'),
            stderr.decode('utf-8', errors='ignore')
        )

    def has_changes(self) -> bool:
        """Проверяет наличие изменений в репозитории."""
        returncode, stdout, _ = self.run_git_command(['git', 'diff', 'dev'])
        return bool(stdout.strip())

    def commit_changes(self) -> bool:
        """Создает коммит с текущими изменениями."""
        try:
            # Проверяем наличие больших файлов
            large_files = self.check_large_files()
            if large_files:
                logging.warning(f"Обнаружены файлы больше {self.max_file_size/1024/1024}MB:")
                for file_path, size in large_files:
                    logging.warning(f"- {file_path}: {size/1024/1024:.2f}MB")
                return False

            # Добавляем все изменения, исключая файлы из Experiments
            _, stdout, _ = self.run_git_command(['git', 'status', '--porcelain'])
            for line in stdout.split('\n'):
                if not line.strip():
                    continue
                file_path = line[3:].strip()
                if not self.should_exclude_file(file_path):
                    self.run_git_command(['git', 'add', file_path])

            # Создаем коммит
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            commit_msg = f"Auto commit: {timestamp}"
            returncode, _, stderr = self.run_git_command(['git', 'commit', '-m', commit_msg])

            if returncode != 0:
                logging.error(f"Ошибка при создании коммита: {stderr}")
                return False

            # Пушим изменения
            returncode, _, stderr = self.run_git_command(['git', 'push', 'origin', 'dev'])
            if returncode != 0:
                logging.error(f"Ошибка при push: {stderr}")
                return False

            logging.info(f"Успешно создан и отправлен коммит: {commit_msg}")
            return True

        except Exception as e:
            logging.error(f"Ошибка при коммите изменений: {str(e)}\n{traceback.format_exc()}")
            return False

    def check_large_files(self) -> list:
        """Проверяет наличие измененных файлов больше max_file_size."""
        _, stdout, _ = self.run_git_command(['git', 'status', '--porcelain'])
        large_files = []

        for line in stdout.split('\n'):
            if not line.strip():
                continue

            # Получаем имя файла из вывода git status
            file_path = line[3:].strip()
            if os.path.exists(self.repo_path / file_path):
                size = os.path.getsize(self.repo_path / file_path)
                if size > self.max_file_size:
                    large_files.append((file_path, size))

        return large_files

def main():
    # Путь к вашему репозиторию
    REPO_PATH = "/home/arthur/xray_ml/github_actual_xrd_recon/xrd_phase_ml"  # Замените на путь к вашему репозиторию
    #CHECK_INTERVAL = 3600  # Интервал проверки в секундах (1 час)

    auto_commit = GitAutoCommit(REPO_PATH)

    try:
        if auto_commit.has_changes():
            logging.info("Обнаружены изменения")
            if auto_commit.commit_changes():
                logging.info("Изменения успешно закоммичены и отправлены")
            else:
                logging.warning("Не удалось закоммитить изменения")
            #auto_commit.run_git_command(['git', 'checkout', 'main'])
        else:
            logging.info("Изменений не обнаружено")

    except Exception as e:
        logging.error(f"Произошла ошибка: {str(e)}\n{traceback.format_exc()}")

        #time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()